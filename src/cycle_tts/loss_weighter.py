"""Meta-learned loss-weighting network φ.

Maps  ([L_spk, L_spec, L_f0, L_id, L_intel],  e_L1,  e_L2)  →  w ∈ Δ^4
(non-negative weights summing to one).

The softmax output is *floored* by ``weight_floor`` and re-normalised to avoid
any single loss collapsing to zero during meta-training (which would prevent the
corresponding gradient signal from ever flowing again).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossWeighterConfig


class LossWeighter(nn.Module):
    def __init__(self, cfg: LossWeighterConfig):
        super().__init__()
        self.cfg = cfg

        # Per-language learnable embeddings.
        self.lang_embed = nn.Embedding(cfg.n_languages, cfg.lang_emb_dim)
        nn.init.normal_(self.lang_embed.weight, std=0.02)

        in_dim = cfg.n_losses + 2 * cfg.lang_emb_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(cfg.n_layers):
            layers.append(nn.Linear(prev, cfg.hidden_dim))
            layers.append(nn.GELU())
            prev = cfg.hidden_dim
        layers.append(nn.Linear(prev, cfg.n_losses))
        self.mlp = nn.Sequential(*layers)

        # Initialise so the first-iteration weights are roughly uniform.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        losses: torch.Tensor,  # [n_losses]   (scalar values, detached recommended)
        lang_l1: int,
        lang_l2: int,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = losses.device
        e1 = self.lang_embed(torch.tensor(lang_l1, device=device))
        e2 = self.lang_embed(torch.tensor(lang_l2, device=device))
        x = torch.cat([losses.detach(), e1, e2], dim=-1)
        logits = self.mlp(x)
        if active_mask is None:
            active = torch.ones_like(losses, dtype=torch.bool, device=device)
        else:
            active = active_mask.to(device=device, dtype=torch.bool)
            if not bool(active.any()):
                active = torch.ones_like(losses, dtype=torch.bool, device=device)
        if self.cfg.use_softmax:
            logits = logits.masked_fill(~active, -1e9)
            w = F.softmax(logits, dim=-1)
        else:
            w = F.softplus(logits) * active.to(logits.dtype)
            w = w / (w.sum() + 1e-8)
        # Floor + renormalise over active losses only.  Inactive/masked losses
        # must never absorb probability mass, otherwise φ can minimize the
        # inner loss by assigning weight to a disabled zero-valued objective.
        w = w + active.to(w.dtype) * self.cfg.weight_floor
        w = w / w.sum()
        return w  # [n_losses]
