"""Meta-learned optimizer ψ.

Implements

    Δθ = ψ(log|g| + ε, sign(g), L_total, k)

applied coordinate-wise with weight sharing across all adapter parameters
(Andrychowicz et al., 2016 — "Learning to learn by gradient descent by gradient
descent").  ``ψ`` is a tiny MLP that maps a per-coordinate feature vector to a
per-coordinate update; the same MLP is reused across all parameters.

For numerical stability we operate on:

    feat = [ log(|g| + ε),  sign(g),  L_total (broadcast),  PE(k) ]

then run a 2-layer MLP and *scale* the output by ``output_scale`` so the very
first inner-loop step makes a small update even when ψ is randomly initialised.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn

from .config import MetaOptConfig


class LearnedOptimizer(nn.Module):
    """Coordinate-wise learned optimizer ψ shared across all adapter parameters."""

    def __init__(self, cfg: MetaOptConfig):
        super().__init__()
        self.cfg = cfg

        # Feature dimensions:
        #   log|g|       : 1   (if use_log_grad)
        #   sign(g)      : 1   (if use_grad_sign)
        #   loss         : 1   (if use_loss_input)
        #   step PE      : max_step_pe   (if use_step_input — sinusoidal)
        in_dim = 0
        if cfg.use_log_grad:
            in_dim += 1
        if cfg.use_grad_sign:
            in_dim += 1
        if cfg.use_loss_input:
            in_dim += 1
        if cfg.use_step_input:
            in_dim += cfg.max_step_pe
        self.in_dim = in_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(cfg.n_layers):
            layers.append(nn.Linear(prev, cfg.hidden_dim))
            layers.append(nn.GELU())
            prev = cfg.hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

        # Initialise the final layer near-zero so initial updates are small.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Precompute step PE basis frequencies.
        if cfg.use_step_input:
            half = cfg.max_step_pe // 2
            div = torch.exp(torch.arange(half) * (-math.log(10_000.0) / max(half - 1, 1)))
            self.register_buffer("pe_div", div, persistent=False)

        self.eps = 1e-8
        self.output_scale = cfg.output_scale

    def _build_step_pe(self, step: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cfg = self.cfg
        if not cfg.use_step_input:
            return torch.empty(0, device=device, dtype=dtype)
        half = cfg.max_step_pe // 2
        s = torch.tensor(float(step), device=device, dtype=dtype)
        args = s * self.pe_div.to(dtype=dtype, device=device)
        pe = torch.empty(cfg.max_step_pe, device=device, dtype=dtype)
        pe[:half] = torch.sin(args)
        pe[half:] = torch.cos(args)
        return pe  # [max_step_pe]

    def compute_update(
        self,
        grads: Sequence[torch.Tensor],
        loss: torch.Tensor,
        step: int,
    ) -> List[torch.Tensor]:
        """Compute per-parameter additive updates Δθ_i for each gradient tensor.

        Parameters
        ----------
        grads : list of tensors, one per adapter parameter, matching its shape.
        loss  : scalar tensor (current L_total).
        step  : int (current inner-loop step index, 0-based).
        """
        cfg = self.cfg
        device = grads[0].device
        # Always run ψ in fp32 for numerical stability; cast back to the input
        # gradient's dtype on the way out so downstream parameter updates stay
        # in the same precision as the adapter (which may be bf16).
        compute_dtype = torch.float32
        out_dtype = grads[0].dtype

        step_pe = self._build_step_pe(step, device=device, dtype=compute_dtype)
        loss_scalar = loss.detach().to(dtype=compute_dtype, device=device).reshape(()).clamp(max=1e3)

        updates: List[torch.Tensor] = []
        for g in grads:
            g_fp = g.detach().to(compute_dtype)
            flat = g_fp.reshape(-1)
            n = flat.numel()
            feats: list[torch.Tensor] = []
            if cfg.use_log_grad:
                feats.append(torch.log(flat.abs() + self.eps).unsqueeze(-1))
            if cfg.use_grad_sign:
                feats.append(torch.sign(flat).unsqueeze(-1))
            if cfg.use_loss_input:
                feats.append(loss_scalar.expand(n).unsqueeze(-1))
            if cfg.use_step_input:
                feats.append(step_pe.unsqueeze(0).expand(n, -1))
            x = torch.cat(feats, dim=-1)  # [n, in_dim]
            delta = self.mlp(x).squeeze(-1)  # [n]
            delta = delta * self.output_scale
            updates.append(delta.reshape_as(g_fp).to(out_dtype))
        return updates

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
