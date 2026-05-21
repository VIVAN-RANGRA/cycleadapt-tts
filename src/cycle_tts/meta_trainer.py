"""Meta-training engine for CycleAdapt-TTS.

Implements Algorithm 1 from the plan, with the speed-ups from REVIEW §2:

  * **Reduced-NFE inner loop**: forward generation uses NFE=6 and the cycle
    reconstruction uses NFE=4 (vs. 32 at evaluation time).
  * **Truncated cycle gradient**: ``ŷ`` is ``.detach()``-ed before being used
    as the prompt of the cycle call, so gradients only flow through the
    second F5-TTS call's adapter parameters.
  * **Functional parameter substitution**: the inner-loop forward calls use
    :func:`torch.func.functional_call` with overridden LoRA tensors, letting
    the autograd graph capture ψ and φ's contributions.
  * **First-order outer updates** (FOMAML for ψ/φ + Reptile for θ₀).

The outer loop's gradient computation:
  - We compute L_outer(θ_K) where θ_K depends on ψ, φ through the inner-loop
    chain of additive updates.
  - ``torch.autograd.grad(L_outer, [ψ.params, φ.params], create_graph=False)``
    yields first-order gradients (the chain through ``g_k = ∇_θ L_inner`` is
    detached because we always pass ``g_k.detach()`` into ψ's input).
  - θ₀ uses Reptile: ``θ₀ ← θ₀ + β · (θ_K - θ₀)``, no autograd needed.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

from .config import CycleAdaptConfig
from .data import ManifestIndex
from .episode_sampler import Episode, EpisodeSampler
from .f5_wrapper import F5CycleWrapper
from .feature_extractors import F0Extractor, MelSpecExtractor, WavLMSpeakerEncoder
from .iaa import IdentityAlignmentAdapter, LoRALayer
from .loss_weighter import LossWeighter
from .losses import (
    LossBundle,
    compute_loss_bundle,
    outer_loss,
)
from .lora_stability import (
    anchor_init_toward_pristine,
    bound_psi_updates,
    clamp_init_params,
    clamp_lora_lists,
    snapshot_pristine_init,
)
from .meta_optimizer import LearnedOptimizer


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: collect LoRA override dict for functional_call
# ---------------------------------------------------------------------------

def _lora_param_name_to_path(iaa: IdentityAlignmentAdapter) -> Dict[int, Tuple[str, str]]:
    """Map index-in-adapter_parameters() -> (param_name_for_A, param_name_for_B).

    Since :meth:`IdentityAlignmentAdapter.adapter_parameters` interleaves A then
    B per record, even indices are A's and odd indices are B's.  We need the
    fully-qualified path inside :class:`f5_tts.model.backbones.dit.DiT` so we
    can pass overrides to :func:`torch.func.functional_call`.
    """
    paths: Dict[int, Tuple[str, str]] = {}
    for i, r in enumerate(iaa.records):
        # The DiT module sees the LoRA layer in place of the original linear,
        # so the parameter path is
        #   transformer_blocks.<blk>.<submodule_path>.A
        # where submodule_path is e.g. "attn.to_q" (or "attn.to_out.0").
        base = f"transformer_blocks.{r.block_idx}.{r.submodule_path}"
        paths[i] = (f"{base}.A", f"{base}.B")
    return paths


def _make_override(
    iaa: IdentityAlignmentAdapter,
    new_A_list: List[torch.Tensor],
    new_B_list: List[torch.Tensor],
    paths: Dict[int, Tuple[str, str]],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for i, _ in enumerate(iaa.records):
        a_path, b_path = paths[i]
        out[a_path] = new_A_list[i]
        out[b_path] = new_B_list[i]
    return out


# ---------------------------------------------------------------------------
# Inner-loop machinery
# ---------------------------------------------------------------------------

@dataclass
class InnerLoopResult:
    final_A: List[torch.Tensor]  # detached final adapter A values
    final_B: List[torch.Tensor]
    last_losses: LossBundle
    last_weights: torch.Tensor
    last_override: Dict[str, torch.Tensor]
    inner_step_losses: List[float]
    # Undetached weighted inner-loss at the FINAL inner step.  Used to compute
    # a *first-order auxiliary gradient* for φ (since FOMAML's ψ pathway
    # ``.detach()``-es the loss before feeding it to the learned optimiser,
    # cutting φ's normal MAML gradient).  Gradient of this scalar w.r.t. φ's
    # parameters is "weighted-sum loss with bundle values held fixed"  — a
    # supervised signal for φ to learn which loss balance minimises the
    # inner-step total.
    last_L_total_for_phi: torch.Tensor


def _inner_loop(
    *,
    episode: Episode,
    iaa: IdentityAlignmentAdapter,
    psi: LearnedOptimizer,
    phi: LossWeighter,
    f5: F5CycleWrapper,
    spk_encoder: WavLMSpeakerEncoder,
    f0_extractor: F0Extractor,
    mel_extractor: MelSpecExtractor,
    cfg: CycleAdaptConfig,
    paths: Dict[int, Tuple[str, str]],
    K: int,
    cycle_grad_truncate: bool = True,
    log_progress: bool = False,
) -> InnerLoopResult:
    """Run K inner-loop steps starting from θ_0 (the meta-learned init).

    Returns the final per-record (A, B) tensors with autograd-tracked dependency
    on ψ, φ (and on θ_0).  These tensors are NOT detached because the outer
    loop will call ``autograd.grad(L_outer(θ_K), [ψ.params, φ.params])``.
    """
    device = next(iaa.dit.parameters()).device
    dtype = next(iaa.dit.parameters()).dtype

    # 1) Start from θ_0 — the meta-learned init.  These are *parameters*
    #    (leaves) but for the inner loop we treat them as the starting values
    #    of a non-leaf chain.  We materialise initial tensors:
    n_rec = len(iaa.records)
    # CRITICAL: never put θ₀ (init_params) directly in the inner-loop graph —
    # autograd would write gradients into θ₀ during inner/outer backprop and
    # Reptile would compound the corruption (root cause of collapsed inference).
    cur_A: List[torch.Tensor] = [
        iaa.init_params[i].detach().to(dtype).clone().requires_grad_(True)
        for i in range(n_rec)
    ]
    cur_B: List[torch.Tensor] = [
        iaa.init_params[n_rec + i].detach().to(dtype).clone().requires_grad_(True)
        for i in range(n_rec)
    ]

    inner_losses: List[float] = []

    # Cache prompt mel once per episode (inner loop calls generate_diff 2×K times).
    with torch.no_grad():
        cached_cond_mel, cached_ref_frames = f5.cache_prompt_mel(episode.x_wav_24k.to(device))

    for k in range(K):
        # -- Forward generation (autograd-tracked through cur_A, cur_B, ψ, φ).
        override = _make_override(iaa, cur_A, cur_B, paths)
        gen_fwd = f5.generate_diff(
            episode.x_wav_24k,
            episode.t_prime,
            episode.t_support,
            nfe_step=cfg.cycle.nfe_forward_inner,
            cfg_strength=cfg.cycle.cfg_strength_inner,
            sway_sampling_coef=-1.0,
            overrides=override,
            seed=k * 31 + 7,  # reproducible per step
            cached_cond_mel=cached_cond_mel,
            cached_ref_frames=cached_ref_frames,
        )
        y_hat = gen_fwd.wave  # [1, T_samples_24k]

        # -- Cycle generation: use ŷ as the prompt for an L1 generation.
        if cycle_grad_truncate:
            y_hat_for_cycle = y_hat.detach()
        else:
            y_hat_for_cycle = y_hat

        gen_cyc = f5.generate_diff(
            y_hat_for_cycle.squeeze(0),
            episode.t_support,  # the "ref text" for the cycle is the L2 text we just generated
            episode.t_prime,    # we want to reconstruct the L1 content
            nfe_step=cfg.cycle.nfe_cycle_inner,
            cfg_strength=cfg.cycle.cfg_strength_inner,
            sway_sampling_coef=-1.0,
            overrides=override,
            seed=k * 17 + 3,
            cached_cond_mel=None,
            cached_ref_frames=None,
        )
        y_hat_cycle = gen_cyc.wave  # [1, T]

        # -- Compute the 5 sub-losses.
        bundle = compute_loss_bundle(
            x_prompt_24k=episode.x_wav_24k.to(device),
            y_hat_24k=y_hat,
            y_hat_cycle_24k=y_hat_cycle,
            spk_encoder=spk_encoder,
            f0_extractor=f0_extractor,
            mel_extractor=mel_extractor,
        )

        # -- Apply the inner-loop mask (skip L_intel, which is non-differentiable).
        lam = cfg.cycle.inner_loss_lambda
        mask = torch.tensor(
            [lam["spk"], lam["spec"], lam["f0"], lam["id"], lam["intel"]],
            device=device, dtype=dtype,
        )
        # -- Learnable weights from φ (uses *detached* loss values per FOMAML).
        # Mask inactive losses before softmax so φ cannot collapse onto L_intel=0.
        active_mask = mask > 0
        w = phi(
            bundle.to_vector(),
            episode.lang_l1_idx,
            episode.lang_l2_idx,
            active_mask=active_mask,
        ).to(dtype)
        L_total = bundle.weighted_sum(w, mask=mask)
        inner_losses.append(float(L_total.detach()))

        if log_progress:
            log.info(
                "  inner k=%d: L_spk=%.3f L_spec=%.3f L_f0=%.3f L_id=%.3f  L_tot=%.3f",
                k, bundle.spk.item(), bundle.spec.item(), bundle.f0.item(), bundle.id.item(), L_total.item(),
            )

        # -- Compute gradient of L_total w.r.t. cur_A, cur_B (FOMAML: detached for ψ input).
        # ``retain_graph=True`` keeps the upstream chain (cur_A_k -> ψ -> cur_A_{k-1} -> ...)
        # alive so the outer-loop backward can still propagate through ψ.
        params = cur_A + cur_B
        grads = torch.autograd.grad(
            L_total,
            params,
            create_graph=False,
            retain_graph=(k < K - 1),
            allow_unused=True,
        )
        # Replace ``None`` gradients (parameter not used in this forward) with zeros.
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        # grads_A: first n_rec, grads_B: rest
        grads_A = list(grads[:n_rec])
        grads_B = list(grads[n_rec:])

        # -- Apply the learned optimizer ψ.  ψ's forward is *autograd-tracked*
        # w.r.t. its own params (cur_A_{k+1} = cur_A_k - ψ(g_k.detach(), ...)).
        updates_A = psi.compute_update([g.detach() for g in grads_A], loss=L_total.detach(), step=k)
        updates_B = psi.compute_update([g.detach() for g in grads_B], loss=L_total.detach(), step=k)
        if cfg.cycle.residual_grad_lr > 0:
            lr = cfg.cycle.residual_grad_lr
            updates_A = [u + lr * g.detach().to(dtype=u.dtype) for u, g in zip(updates_A, grads_A)]
            updates_B = [u + lr * g.detach().to(dtype=u.dtype) for u, g in zip(updates_B, grads_B)]
        updates_A = bound_psi_updates(updates_A, cfg.stab)
        updates_B = bound_psi_updates(updates_B, cfg.stab)

        new_A = [a - u for a, u in zip(cur_A, updates_A)]
        new_B = [b - u for b, u in zip(cur_B, updates_B)]
        clamp_lora_lists(new_A, new_B, cfg.stab, cfg.iaa)

        cur_A, cur_B = new_A, new_B

    # Build a φ-only auxiliary scalar: same formula as L_total but with the
    # *bundle values detached* so the only autograd path is φ → w → L.
    detached_bundle = LossBundle(
        spk=bundle.spk.detach(),
        spec=bundle.spec.detach(),
        f0=bundle.f0.detach(),
        id=bundle.id.detach(),
        intel=bundle.intel.detach(),
    )
    L_total_for_phi = detached_bundle.weighted_sum(w, mask=mask)

    return InnerLoopResult(
        final_A=cur_A,
        final_B=cur_B,
        last_losses=bundle,
        last_weights=w.detach(),
        last_override=_make_override(iaa, cur_A, cur_B, paths),
        inner_step_losses=inner_losses,
        last_L_total_for_phi=L_total_for_phi,
    )


# ---------------------------------------------------------------------------
# Meta-trainer
# ---------------------------------------------------------------------------

class MetaTrainer:
    def __init__(
        self,
        cfg: CycleAdaptConfig,
        iaa: IdentityAlignmentAdapter,
        psi: LearnedOptimizer,
        phi: LossWeighter,
        f5: F5CycleWrapper,
        spk_encoder: WavLMSpeakerEncoder,
        f0_extractor: F0Extractor,
        mel_extractor: MelSpecExtractor,
        manifest: ManifestIndex,
        *,
        device: str = "cuda",
        log_wandb: bool = False,
        log_tb: bool = True,
    ):
        self.cfg = cfg
        self.iaa = iaa
        self.psi = psi.to(device)
        self.phi = phi.to(device)
        self.f5 = f5
        self.spk_encoder = spk_encoder
        self.f0_extractor = f0_extractor
        self.mel_extractor = mel_extractor
        self.manifest = manifest
        self.device = device

        self.paths = _lora_param_name_to_path(iaa)

        # Optimizers for ψ and φ — Adam.  Reptile β governs θ_0 update.
        self.psi_opt = torch.optim.Adam(self.psi.parameters(), lr=cfg.train.lr_optim, weight_decay=cfg.train.weight_decay)
        self.phi_opt = torch.optim.Adam(self.phi.parameters(), lr=cfg.train.lr_optim, weight_decay=cfg.train.weight_decay)

        # Logging
        self.log_wandb = log_wandb and wandb is not None
        self.tb_writer = SummaryWriter(str(cfg.log_dir)) if (log_tb and SummaryWriter is not None) else None
        if self.log_wandb:
            wandb.init(project="cycleadapt-tts", name=cfg.run_name, config=asdict(cfg), dir=str(cfg.log_dir))

        self.sampler = EpisodeSampler(cfg, manifest, seed=cfg.train.seed)
        self.global_step = 0
        # Pristine θ₀ snapshot (A∼N(0,σ²), B=0) for anchoring after Reptile.
        self._pristine_init = snapshot_pristine_init(self.iaa)

    # ------------------------------------------------------------------

    def _one_meta_iter(self) -> Dict[str, float]:
        """Run one meta-iteration: B episodes + ψ/φ update + Reptile θ_0 update."""
        cfg = self.cfg

        # Buffers to accumulate
        psi_grads: Optional[List[torch.Tensor]] = None
        phi_grads: Optional[List[torch.Tensor]] = None
        delta_init: List[torch.Tensor] = [torch.zeros_like(p) for p in self.iaa.init_parameters()]

        running_outer = 0.0
        running_inner_first = 0.0
        running_inner_last = 0.0
        n_ok = 0
        episode_diagnostics: List[Dict[str, float]] = []

        for b in range(cfg.train.B):
            episode = self.sampler.sample()
            if episode is None:
                continue

            # ---- Inner loop ----
            inner = _inner_loop(
                episode=episode,
                iaa=self.iaa,
                psi=self.psi,
                phi=self.phi,
                f5=self.f5,
                spk_encoder=self.spk_encoder,
                f0_extractor=self.f0_extractor,
                mel_extractor=self.mel_extractor,
                cfg=cfg,
                paths=self.paths,
                K=cfg.cycle.K_train,
                cycle_grad_truncate=cfg.cycle.truncate_cycle_grad,
                log_progress=False,
            )

            # ---- Outer-loop query generation (reuse prompt mel cache) ----
            with torch.no_grad():
                outer_mel, outer_ref = self.f5.cache_prompt_mel(episode.x_wav_24k.to(self.device))
            gen_q = self.f5.generate_diff(
                episode.x_wav_24k,
                episode.t_prime,
                episode.t_query,
                nfe_step=cfg.cycle.nfe_outer,
                cfg_strength=cfg.cycle.cfg_strength_outer,
                sway_sampling_coef=-1.0,
                overrides=inner.last_override,
                seed=42,
                cached_cond_mel=outer_mel,
                cached_ref_frames=outer_ref,
            )
            y_query = gen_q.wave

            L_outer, outer_diag = outer_loss(
                episode.x_wav_24k.to(self.device),
                y_query,
                spk_encoder=self.spk_encoder,
                f0_extractor=self.f0_extractor,
                lambda_f0=cfg.cycle.outer_lambda_f0,
            )

            if not torch.isfinite(L_outer):
                log.warning("Skipping episode: non-finite L_outer=%s", L_outer.item())
                continue

            # ---- ψ first-order outer gradient (FOMAML) ----
            ep_psi_grads_raw = torch.autograd.grad(
                L_outer,
                list(self.psi.parameters()),
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )
            ep_psi_grads = [g if g is not None else torch.zeros_like(p)
                             for g, p in zip(ep_psi_grads_raw, self.psi.parameters())]

            # ---- φ first-order *auxiliary* gradient ----
            phi_params = [p for p in self.phi.parameters() if p.requires_grad]
            if phi_params:
                ep_phi_grads_raw = torch.autograd.grad(
                    inner.last_L_total_for_phi,
                    phi_params,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )
                ep_phi_grads = []
                gi = 0
                for p in self.phi.parameters():
                    if p.requires_grad:
                        g = ep_phi_grads_raw[gi]
                        ep_phi_grads.append(g if g is not None else torch.zeros_like(p))
                        gi += 1
                    else:
                        ep_phi_grads.append(torch.zeros_like(p))
                del ep_phi_grads_raw
            else:
                ep_phi_grads = [torch.zeros_like(p) for p in self.phi.parameters()]
            if psi_grads is None:
                psi_grads = ep_psi_grads
                phi_grads = ep_phi_grads
            else:
                psi_grads = [a + b for a, b in zip(psi_grads, ep_psi_grads)]
                phi_grads = [a + b for a, b in zip(phi_grads, ep_phi_grads)]
            del ep_psi_grads_raw

            # ---- Reptile delta for θ_0 ----
            # Accumulate (θ_K - θ_0) for both A's and B's.
            n_rec = len(self.iaa.records)
            with torch.no_grad():
                for i in range(n_rec):
                    delta_init[i] += inner.final_A[i].detach() - self.iaa.init_params[i]
                    delta_init[n_rec + i] += inner.final_B[i].detach() - self.iaa.init_params[n_rec + i]

            running_outer += float(L_outer.detach())
            running_inner_first += float(inner.inner_step_losses[0]) if inner.inner_step_losses else 0.0
            running_inner_last += float(inner.inner_step_losses[-1]) if inner.inner_step_losses else 0.0
            n_ok += 1

            episode_diagnostics.append({
                "speaker": episode.speaker_id,
                "lang_pair": f"{episode.lang_l1}->{episode.lang_l2}",
                "L_outer": float(L_outer.detach()),
                "L_inner_first": float(inner.inner_step_losses[0]) if inner.inner_step_losses else float("nan"),
                "L_inner_last": float(inner.inner_step_losses[-1]) if inner.inner_step_losses else float("nan"),
                "w_spk": float(inner.last_weights[0]),
                "w_spec": float(inner.last_weights[1]),
                "w_f0": float(inner.last_weights[2]),
                "w_id": float(inner.last_weights[3]),
                "w_intel": float(inner.last_weights[4]),
            })

        if n_ok == 0:
            return {"L_outer": float("nan"), "n_ok": 0}

        # ---- Apply ψ / φ updates ----
        for p, g in zip(self.psi.parameters(), psi_grads):
            p.grad = g / max(n_ok, 1)
        for p, g in zip(self.phi.parameters(), phi_grads):
            p.grad = g / max(n_ok, 1)
        torch.nn.utils.clip_grad_norm_(self.psi.parameters(), cfg.train.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.phi.parameters(), cfg.train.grad_clip)
        self.psi_opt.step()
        self.phi_opt.step()
        self.psi_opt.zero_grad(set_to_none=True)
        self.phi_opt.zero_grad(set_to_none=True)

        # ---- Apply Reptile update for θ_0 ----
        beta = self.cfg.train.reptile_beta
        n_rec = len(self.iaa.records)
        with torch.no_grad():
            for p, d in zip(self.iaa.init_parameters(), delta_init):
                p.add_(d / max(n_ok, 1), alpha=beta)
            anchor_init_toward_pristine(
                self.iaa.init_params, self._pristine_init, n_rec, cfg.stab.init_anchor_strength,
            )
            clamp_init_params(self.iaa.init_params, n_rec, cfg.stab, cfg.iaa)
            # Keep leaf LoRA in sync for any direct (non-override) forwards.
            self.iaa.reset_from_init()

        out = {
            "L_outer": running_outer / n_ok,
            "L_inner_first": running_inner_first / n_ok,
            "L_inner_last": running_inner_last / n_ok,
            "n_ok": n_ok,
            "episodes": episode_diagnostics,
        }
        for key in ["w_spk", "w_spec", "w_f0", "w_id", "w_intel"]:
            out[key] = sum(e[key] for e in episode_diagnostics) / max(len(episode_diagnostics), 1)
        return out

    # ------------------------------------------------------------------

    def fit(self, *, start_step: int = 0) -> None:
        cfg = self.cfg
        log.info("Starting meta-training: M=%d B=%d K=%d  (iaa params=%d, ψ=%d, φ=%d)  start_step=%d",
                 cfg.train.M, cfg.train.B, cfg.cycle.K_train,
                 self.iaa.n_adapter_params(), self.psi.num_params(),
                 sum(p.numel() for p in self.phi.parameters()),
                 start_step)

        t0 = time.time()
        history: List[Dict] = []

        for m in range(start_step, cfg.train.M):
            self.global_step = m
            t_iter = time.time()
            metrics = self._one_meta_iter()
            iter_time = time.time() - t_iter

            if cfg.train.skip_nan_iters and (
                not math.isfinite(metrics.get("L_outer", float("nan")))
                or metrics.get("n_ok", 0) == 0
            ):
                log.warning("[meta %5d/%d] skipped bad iter (L_out=%s n_ok=%s)",
                            m, cfg.train.M, metrics.get("L_outer"), metrics.get("n_ok"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            history.append({"step": m, "time_s": iter_time, **{k: v for k, v in metrics.items() if k != "episodes"}})

            if m % cfg.train.log_every == 0:
                log.info(
                    "[meta %5d/%d] L_out=%.4f L_in_first=%.4f L_in_last=%.4f "
                    "w=[%.2f %.2f %.2f %.2f %.2f]  (%.1fs/it, ETA %dmin)",
                    m, cfg.train.M, metrics["L_outer"], metrics["L_inner_first"], metrics["L_inner_last"],
                    metrics.get("w_spk", float("nan")), metrics.get("w_spec", float("nan")),
                    metrics.get("w_f0", float("nan")), metrics.get("w_id", float("nan")),
                    metrics.get("w_intel", float("nan")),
                    iter_time, int((cfg.train.M - m) * iter_time / 60),
                )
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("meta/L_outer", metrics["L_outer"], m)
                    self.tb_writer.add_scalar("meta/L_inner_first", metrics["L_inner_first"], m)
                    self.tb_writer.add_scalar("meta/L_inner_last", metrics["L_inner_last"], m)
                    self.tb_writer.add_scalar("meta/iter_time_s", iter_time, m)
                if self.log_wandb:
                    wandb.log({
                        "meta/L_outer": metrics["L_outer"],
                        "meta/L_inner_first": metrics["L_inner_first"],
                        "meta/L_inner_last": metrics["L_inner_last"],
                        "meta/iter_time_s": iter_time,
                    }, step=m)

            if (m + 1) % cfg.train.ckpt_every == 0 or m == cfg.train.M - 1:
                ck_path = cfg.ckpt_dir / f"step{m+1:06d}.pt"
                self.save_checkpoint(ck_path)
                log.info("  saved -> %s", ck_path)

            # Memory hygiene: release the previous iteration's autograd
            # leftovers + force defragmentation every iter.  At ~60 GB peak
            # usage we were running 200 MB short of OOM by step ~660.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final save
        self.save_checkpoint(cfg.ckpt_dir / "final.pt")
        # History dump
        (cfg.log_dir / "meta_history.jsonl").write_text("\n".join(json.dumps(h) for h in history))
        log.info("Meta-training done in %.1fmin", (time.time() - t0) / 60)

    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "iaa": self.iaa.state_dict_iaa_only(),
            "psi": self.psi.state_dict(),
            "phi": self.phi.state_dict(),
            "psi_opt": self.psi_opt.state_dict(),
            "phi_opt": self.phi_opt.state_dict(),
            "step": self.global_step,
            "cfg": {
                "iaa": asdict(self.cfg.iaa),
                "psi": asdict(self.cfg.psi),
                "phi": asdict(self.cfg.phi),
                "cycle": asdict(self.cfg.cycle),
                "train": asdict(self.cfg.train),
            },
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.iaa.load_state_dict_iaa_only(payload["iaa"])
        n_rec = len(self.iaa.records)
        clamp_init_params(self.iaa.init_params, n_rec, self.cfg.stab, self.cfg.iaa)
        self.iaa.reset_from_init()
        self.psi.load_state_dict(payload["psi"])
        self.phi.load_state_dict(payload["phi"])
        if "psi_opt" in payload:
            self.psi_opt.load_state_dict(payload["psi_opt"])
            self.phi_opt.load_state_dict(payload["phi_opt"])
        self.global_step = payload.get("step", 0)
