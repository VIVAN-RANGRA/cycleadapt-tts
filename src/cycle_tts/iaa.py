"""Identity Alignment Adapter (IAA).

LoRA-style low-rank adapters injected into the speaker-conditioning pathway of
F5-TTS.  Each ``LoRALayer`` wraps a frozen :class:`nn.Linear` and adds
``Δh = (alpha / r) · x A^T B^T``  where  ``A ∈ R^{r×d_in}``,  ``B ∈ R^{d_out×r}``.

Initialization (per the plan §4.2):
  * ``A ∼ N(0, σ²)``  with ``σ = init_std`` (default 0.02)
  * ``B = 0``  → ``ΔW = 0`` at start, so the adapter starts as identity.

A second set of parameters ``A_0``, ``B_0`` (registered as buffers seeded from a
``θ_0`` slot) lets meta-training learn the *initialization* the inner-loop
starts from while the inner-loop variables ``A``, ``B`` keep their leaf-status
for cheap autograd.

Injection target (F5-TTS v1 Base, ``depth=22``) — the top ``n_top_blocks``
``DiTBlock`` s wrap:

  * ``attn.to_q``, ``attn.to_k``, ``attn.to_v``, ``attn.to_out[0]``    (attention)
  * ``attn_norm.linear``                                               (adaLN modulation)

We do not touch the FF layers because they carry general content, not speaker
identity, and adding LoRA there bloats parameter count.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import IAAConfig


# ---------------------------------------------------------------------------
# Core LoRA layer
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """Wraps a frozen ``nn.Linear`` with an additive low-rank update."""

    base: nn.Linear

    def __init__(self, base: nn.Linear, rank: int, alpha: float, init_std: float = 0.02):
        super().__init__()
        # Store base as a non-module attribute so that gradients don't even
        # try to flow into the frozen parameters when we list ``adapter_parameters``.
        # We add it as a regular submodule too so that ``state_dict()`` and
        # ``.to(device)`` work transparently.
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        d_in = base.in_features
        d_out = base.out_features

        # Inner-loop trainable params (re-initialised from θ_0 every episode).
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.empty(d_out, rank))

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.init_std = init_std

        self.reset_to_pristine()

    def reset_to_pristine(self) -> None:
        """Re-initialise A∼N(0,σ²), B=0 (pre-meta-training default)."""
        nn.init.normal_(self.A, mean=0.0, std=self.init_std)
        nn.init.zeros_(self.B)

    @torch.no_grad()
    def copy_from(self, A0: torch.Tensor, B0: torch.Tensor) -> None:
        """Copy meta-learned initialization θ₀ = (A₀, B₀) into the leaf params."""
        self.A.copy_(A0)
        self.B.copy_(B0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # ΔW · x with broadcasting over the leading dims:
        #   x         : [..., d_in]
        #   x A^T     : [..., r]
        #   (x A^T) B^T : [..., d_out]
        lora_out = (x @ self.A.t()) @ self.B.t()
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, in={self.base.in_features}, out={self.base.out_features}"


# ---------------------------------------------------------------------------
# Bookkeeping: which submodules in F5-TTS receive an adapter
# ---------------------------------------------------------------------------

@dataclass
class InjectionRecord:
    """Where one LoRA layer lives within the DiT module graph."""

    block_idx: int
    submodule_path: str   # e.g. "attn.to_q", "attn_norm.linear"
    lora: LoRALayer


def _resolve_targets(target_modules: str) -> List[str]:
    """Translate the short config name into a list of dotted paths inside a DiTBlock."""
    mapping = {
        "qv":       ["attn.to_q", "attn.to_v"],
        "qkv":      ["attn.to_q", "attn.to_k", "attn.to_v"],
        "qkvo":     ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"],
        "qkvo_ada": ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn_norm.linear"],
    }
    if target_modules not in mapping:
        raise ValueError(f"Unknown target_modules: {target_modules!r}")
    return mapping[target_modules]


def _get_submodule(root: nn.Module, dotted_path: str) -> nn.Module:
    cur = root
    for part in dotted_path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _set_submodule(root: nn.Module, dotted_path: str, new_module: nn.Module) -> None:
    parts = dotted_path.split(".")
    cur = root
    for part in parts[:-1]:
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    last = parts[-1]
    if last.isdigit():
        cur[int(last)] = new_module
    else:
        setattr(cur, last, new_module)


# ---------------------------------------------------------------------------
# IdentityAlignmentAdapter — the user-facing wrapper
# ---------------------------------------------------------------------------

class IdentityAlignmentAdapter(nn.Module):
    """Injects LoRA layers into a frozen DiT and exposes them as a flat parameter set.

    Lifecycle:
        iaa = IdentityAlignmentAdapter(cfg, dit)
        iaa.freeze_base()                # ensure DiT params don't get gradients
        iaa.reset_from_init()            # copy θ₀ → leaf params at start of episode
        ...inner-loop steps that update iaa.adapter_parameters() in-place...
        iaa.flatten_params()             # access flat θ_K for Reptile update
    """

    def __init__(self, cfg: IAAConfig, dit: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.dit = dit

        # Determine which DiT blocks to adapt.
        depth = len(dit.transformer_blocks)
        if cfg.n_top_blocks > depth:
            raise ValueError(f"n_top_blocks={cfg.n_top_blocks} > DiT depth={depth}")
        self.adapted_block_idxs = list(range(depth - cfg.n_top_blocks, depth))

        targets = _resolve_targets(cfg.target_modules)

        # Inject LoRA layers.  ``base`` may live in bf16 if the DiT was cast;
        # we match its dtype on the LoRA leaves so that forward passes without
        # ``functional_call`` overrides still work.  The meta-learned init
        # buffer ``init_params`` is kept in fp32 because Adam-style outer
        # updates are noisy in bf16.
        self.records: List[InjectionRecord] = []
        for blk_idx in self.adapted_block_idxs:
            block = dit.transformer_blocks[blk_idx]
            for path in targets:
                base = _get_submodule(block, path)
                if not isinstance(base, nn.Linear):
                    raise TypeError(f"Target {path!r} in block {blk_idx} is not nn.Linear: {type(base).__name__}")
                lora = LoRALayer(base, rank=cfg.rank, alpha=cfg.alpha, init_std=cfg.init_std)
                # Match the dtype/device of the base linear for direct (no-override) forwards.
                lora.A.data = lora.A.data.to(dtype=base.weight.dtype, device=base.weight.device)
                lora.B.data = lora.B.data.to(dtype=base.weight.dtype, device=base.weight.device)
                _set_submodule(block, path, lora)
                self.records.append(InjectionRecord(blk_idx, path, lora))

        # ParameterList of θ_0 (the meta-learned initialization), kept in fp32 for
        # numerically stable outer-loop / Adam updates (LoRA A/B leaves may be bf16).
        self.init_params = nn.ParameterList([
            nn.Parameter(torch.empty(r.lora.A.shape, dtype=torch.float32, device=r.lora.A.device))
            for r in self.records
        ] + [
            nn.Parameter(torch.empty(r.lora.B.shape, dtype=torch.float32, device=r.lora.B.device))
            for r in self.records
        ])
        # Seed θ_0 from the pristine LoRA init so the very first meta-iter is sane.
        with torch.no_grad():
            for idx, r in enumerate(self.records):
                self.init_params[idx].copy_(r.lora.A.float())
                self.init_params[len(self.records) + idx].copy_(r.lora.B.float())

        self.freeze_base()

    # -- API --------------------------------------------------------------

    def freeze_base(self) -> None:
        """Set ``requires_grad=False`` on every DiT parameter except adapters."""
        for n, p in self.dit.named_parameters():
            # Adapter params live under ".A" / ".B" inside LoRALayer.
            if n.endswith(".A") or n.endswith(".B"):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def reset_from_init(self) -> None:
        """Copy meta-learned θ_0 into the inner-loop leaf parameters (A, B)."""
        n_rec = len(self.records)
        with torch.no_grad():
            for idx, r in enumerate(self.records):
                r.lora.A.copy_(self.init_params[idx])
                r.lora.B.copy_(self.init_params[n_rec + idx])

    def adapter_parameters(self) -> List[torch.nn.Parameter]:
        """All inner-loop trainable adapter parameters (the LoRA A/B leaves)."""
        params: List[torch.nn.Parameter] = []
        for r in self.records:
            params.append(r.lora.A)
            params.append(r.lora.B)
        return params

    def init_parameters(self) -> List[torch.nn.Parameter]:
        """The meta-learned init θ_0 (used by Reptile / outer optimizer)."""
        return list(self.init_params)

    def n_adapter_params(self) -> int:
        return sum(p.numel() for p in self.adapter_parameters())

    @torch.no_grad()
    def flatten_adapter(self) -> torch.Tensor:
        """Flatten current (A, B) leaves into a single 1-D vector (for diagnostics)."""
        return torch.cat([p.detach().flatten() for p in self.adapter_parameters()])

    @torch.no_grad()
    def flatten_init(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in self.init_parameters()])

    def state_dict_iaa_only(self) -> Dict[str, torch.Tensor]:
        """Return only IAA-related tensors, suitable for small checkpoints."""
        n_rec = len(self.records)
        sd: Dict[str, torch.Tensor] = {}
        for idx, r in enumerate(self.records):
            key = f"block{r.block_idx:02d}__{r.submodule_path.replace('.', '_')}"
            sd[f"{key}.A"] = r.lora.A.detach().cpu()
            sd[f"{key}.B"] = r.lora.B.detach().cpu()
            sd[f"{key}.A0"] = self.init_params[idx].detach().cpu()
            sd[f"{key}.B0"] = self.init_params[n_rec + idx].detach().cpu()
        return sd

    def load_state_dict_iaa_only(self, sd: Dict[str, torch.Tensor]) -> None:
        n_rec = len(self.records)
        for idx, r in enumerate(self.records):
            key = f"block{r.block_idx:02d}__{r.submodule_path.replace('.', '_')}"
            # Prefer A0/B0 (meta-learned θ₀); fall back to A/B for older checkpoints.
            a0 = sd.get(f"{key}.A0", sd[f"{key}.A"])
            b0 = sd.get(f"{key}.B0", sd[f"{key}.B"])
            self.init_params[idx].data.copy_(a0)
            self.init_params[n_rec + idx].data.copy_(b0)
        self.reset_from_init()
