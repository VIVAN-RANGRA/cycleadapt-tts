"""LoRA stability utilities for CycleAdapt-TTS.

Meta-training can push θ₀ (LoRA A/B) to magnitudes that dominate the frozen
F5 backbone (Δh = (α/r) · x Aᵀ Bᵀ), producing near-silent or garbage audio at
inference.  These helpers:

  * clamp adapter tensors to safe ranges,
  * anchor θ₀ toward the pristine (A∼N(0,σ²), B=0) initialization,
  * bound learned-optimizer step sizes.

Used in both training (inner loop + Reptile) and inference (load + TTT).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from .config import IAAConfig, StabilityConfig


def clamp_lora_pair(
    A: torch.Tensor,
    B: torch.Tensor,
    stab: StabilityConfig,
    *,
    init_std: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In-place clamp one (A, B) LoRA pair.  Returns (A, B) for chaining."""
    with torch.no_grad():
        A.clamp_(-stab.max_A_abs, stab.max_A_abs)
        B.clamp_(-stab.max_B_abs, stab.max_B_abs)
        # Optional soft cap on effective rank-1 scale: ‖A‖_F · ‖B‖_F
        eff = A.float().norm() * B.float().norm()
        if eff > stab.max_AB_product_norm and eff > 1e-8:
            scale = (stab.max_AB_product_norm / eff) ** 0.5
            A.mul_(scale)
            B.mul_(scale)
    return A, B


def clamp_lora_lists(
    A_list: Sequence[torch.Tensor],
    B_list: Sequence[torch.Tensor],
    stab: StabilityConfig,
    iaa_cfg: IAAConfig,
) -> None:
    for A, B in zip(A_list, B_list):
        clamp_lora_pair(A, B, stab, init_std=iaa_cfg.init_std)


def clamp_init_params(
    init_params: torch.nn.ParameterList,
    n_rec: int,
    stab: StabilityConfig,
    iaa_cfg: IAAConfig,
) -> None:
    """Clamp all θ₀ slots in the ParameterList (first n_rec = A, rest = B)."""
    for i in range(n_rec):
        clamp_lora_pair(init_params[i], init_params[n_rec + i], stab, init_std=iaa_cfg.init_std)


def anchor_init_toward_pristine(
    init_params: torch.nn.ParameterList,
    pristine: Sequence[torch.Tensor],
    n_rec: int,
    strength: float,
) -> None:
    """θ₀ ← (1 - λ) θ₀ + λ θ_pristine  (keeps adapters near identity start)."""
    if strength <= 0:
        return
    with torch.no_grad():
        for i, p in enumerate(init_params):
            p.mul_(1.0 - strength).add_(pristine[i], alpha=strength)


def bound_psi_updates(
    updates: List[torch.Tensor],
    stab: StabilityConfig,
) -> List[torch.Tensor]:
    """Clip each ψ-produced update tensor by global norm."""
    if stab.max_psi_update_norm <= 0:
        return updates
    out = []
    for u in updates:
        n = u.float().norm()
        if n > stab.max_psi_update_norm:
            u = u * (stab.max_psi_update_norm / (n + 1e-8))
        out.append(u)
    return out


def snapshot_pristine_init(iaa) -> List[torch.Tensor]:
    """Clone current θ₀ right after IAA construction (pristine A,B)."""
    return [p.detach().clone() for p in iaa.init_parameters()]
