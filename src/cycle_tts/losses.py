"""Cycle-consistent and forward losses for CycleAdapt-TTS.

All inner-loop losses are differentiable in the generated waveform.  They take
*waveforms at 24 kHz* as input and internally resample / featurise as needed.

Sub-losses
----------
``L_spk``   cycle speaker-cosine:    1 - cos(SpkEnc(x), SpkEnc(ŷ'))
``L_spec``  cycle mel L1 (DTW-aligned by zero-pad/truncate)
``L_f0``    cycle F0 Pearson:        1 - PCC(F0(x), F0(ŷ')) on voiced frames
``L_id``    forward speaker-cosine:  1 - cos(SpkEnc(x), SpkEnc(ŷ))
``L_intel`` ASR-CER on ŷ (NaN at train time — evaluated only at outer/eval)

The outer-loop loss bundles SIM-o + λ·F0_PCC; CER is reported alongside but is
not included in the gradient because it is non-differentiable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .feature_extractors import (
    F0Extractor,
    MelSpecExtractor,
    WavLMSpeakerEncoder,
    cosine_sim,
    resample,
)


# ---------------------------------------------------------------------------
# Loss bundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class LossBundle:
    spk: torch.Tensor          # scalar tensor
    spec: torch.Tensor
    f0: torch.Tensor
    id: torch.Tensor
    intel: torch.Tensor

    def to_vector(self) -> torch.Tensor:
        return torch.stack([self.spk, self.spec, self.f0, self.id, self.intel])

    def weighted_sum(self, w: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """w: [5] weights. Optional mask gives per-loss lambda/active status.

        The effective weights are renormalised after applying ``mask``.  Without
        this, a learned weighter can hide probability mass on disabled losses
        (notably the non-differentiable intelligibility slot) and shrink the
        inner-loop gradient toward zero.
        """
        v = self.to_vector()
        if mask is not None:
            eff_w = w * mask.to(dtype=w.dtype, device=w.device)
            eff_w = eff_w / eff_w.sum().clamp(min=1e-8)
            return (eff_w * v).sum()
        return (w * v).sum()


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _safe_pearson(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pearson correlation, returning 0 if degenerate (all-equal / all-masked)."""
    if mask is not None:
        x = x[mask]
        y = y[mask]
    if x.numel() < 4:
        return x.new_zeros(())
    xm = x - x.mean()
    ym = y - y.mean()
    num = (xm * ym).sum()
    den = (xm.pow(2).sum().sqrt() * ym.pow(2).sum().sqrt()).clamp(min=1e-8)
    return num / den


def f0_pcc_loss(
    wav_a_24k: torch.Tensor,
    wav_b_24k: torch.Tensor,
    f0_extractor: F0Extractor,
    log_hz: bool = True,
    voicing_thresh: float = 0.5,
) -> torch.Tensor:
    """1 - PCC(F0(a), F0(b)) on jointly voiced frames after length matching."""
    wav_a_24k = _cap_for_loss(wav_a_24k)
    wav_b_24k = _cap_for_loss(wav_b_24k)
    a16 = resample(wav_a_24k, 24_000, f0_extractor.target_sr)
    b16 = resample(wav_b_24k, 24_000, f0_extractor.target_sr)
    f0_a, vp_a = f0_extractor(a16)
    f0_b, vp_b = f0_extractor(b16)

    # Length match by truncation (DTW alignment is overkill for short clips).
    n = min(f0_a.shape[-1], f0_b.shape[-1])
    f0_a = f0_a[..., :n].squeeze(0)
    f0_b = f0_b[..., :n].squeeze(0)
    vp_a = vp_a[..., :n].squeeze(0)
    vp_b = vp_b[..., :n].squeeze(0)

    voiced = (vp_a > voicing_thresh) & (vp_b > voicing_thresh)
    if voiced.sum() < 8:
        return f0_a.new_tensor(1.0)

    if log_hz:
        f0_a = torch.log(f0_a.clamp(min=1.0))
        f0_b = torch.log(f0_b.clamp(min=1.0))

    pcc = _safe_pearson(f0_a, f0_b, mask=voiced)
    return 1.0 - pcc


def mel_l1_loss(
    wav_a_24k: torch.Tensor,
    wav_b_24k: torch.Tensor,
    mel_extractor: MelSpecExtractor,
) -> torch.Tensor:
    """Length-matched mel L1 distance."""
    wav_a_24k = _cap_for_loss(wav_a_24k)
    wav_b_24k = _cap_for_loss(wav_b_24k)
    mel_a = mel_extractor(wav_a_24k)  # [B, n_mel, T]
    mel_b = mel_extractor(wav_b_24k)
    n = min(mel_a.shape[-1], mel_b.shape[-1])
    return F.l1_loss(mel_a[..., :n], mel_b[..., :n])


# Maximum audio length (in seconds at 24 kHz) fed into WavLM / F0 for the
# loss computation.  WavLM activations scale with audio length, and at
# 30+ seconds the cached forward graph blows past 70 GB on a single A100.
# 12 s is more than enough for a stable speaker embedding (state-of-the-art
# speaker verification systems typically use 3-10 s segments).
_MAX_LOSS_AUDIO_SEC = 12.0
_MAX_LOSS_SAMPLES_24K = int(_MAX_LOSS_AUDIO_SEC * 24_000)


def _cap_for_loss(wav_24k: torch.Tensor, *, max_samples: int = _MAX_LOSS_SAMPLES_24K) -> torch.Tensor:
    """Center-crop a 24 kHz waveform to ``max_samples`` if it is longer.

    Preserves autograd graph (uses indexing, no detach).
    """
    if wav_24k.shape[-1] <= max_samples:
        return wav_24k
    excess = wav_24k.shape[-1] - max_samples
    start = excess // 2
    return wav_24k[..., start:start + max_samples]


def speaker_cosine_loss(
    wav_a_24k: torch.Tensor,
    wav_b_24k: torch.Tensor,
    spk_encoder: WavLMSpeakerEncoder,
) -> torch.Tensor:
    # Cap both inputs to ``_MAX_LOSS_AUDIO_SEC`` BEFORE resample so we never
    # build a >70 GB activation graph through WavLM.
    wav_a_24k = _cap_for_loss(wav_a_24k)
    wav_b_24k = _cap_for_loss(wav_b_24k)
    a16 = resample(wav_a_24k, 24_000, spk_encoder.target_sr)
    b16 = resample(wav_b_24k, 24_000, spk_encoder.target_sr)
    e_a = spk_encoder(a16)
    e_b = spk_encoder(b16)
    sim = cosine_sim(e_a, e_b).mean()
    return 1.0 - sim


# ---------------------------------------------------------------------------
# Full bundle
# ---------------------------------------------------------------------------

def embed_prompt_24k(
    x_prompt_24k: torch.Tensor,
    spk_encoder: WavLMSpeakerEncoder,
) -> torch.Tensor:
    """L2-normalised WavLM embedding for the (fixed) prompt; reuse across TTT steps."""
    wav = _cap_for_loss(x_prompt_24k)
    a16 = resample(wav, 24_000, spk_encoder.target_sr)
    with torch.no_grad():
        return spk_encoder(a16).detach()


def speaker_cosine_loss_from_embed(
    e_prompt: torch.Tensor,
    wav_b_24k: torch.Tensor,
    spk_encoder: WavLMSpeakerEncoder,
) -> torch.Tensor:
    """``1 - cos(e_prompt, SpkEnc(wav_b))`` — prompt embed is constant per item."""
    wav_b_24k = _cap_for_loss(wav_b_24k)
    b16 = resample(wav_b_24k, 24_000, spk_encoder.target_sr)
    e_b = spk_encoder(b16)
    sim = cosine_sim(e_prompt, e_b).mean()
    return 1.0 - sim


def compute_loss_bundle(
    x_prompt_24k: torch.Tensor,      # original prompt (no grad needed)
    y_hat_24k: torch.Tensor,         # forward generation in L2 (grad-tracked)
    y_hat_cycle_24k: torch.Tensor,   # cycle reconstruction in L1 (grad-tracked)
    *,
    spk_encoder: WavLMSpeakerEncoder,
    f0_extractor: F0Extractor,
    mel_extractor: MelSpecExtractor,
    include_intel: bool = False,
    e_prompt: Optional[torch.Tensor] = None,
    id_only: bool = False,
) -> LossBundle:
    """Compute the 5 sub-losses described in §5.2 of the plan."""
    if e_prompt is None:
        e_prompt = embed_prompt_24k(x_prompt_24k, spk_encoder)

    if id_only:
        L_id = speaker_cosine_loss_from_embed(e_prompt, y_hat_24k, spk_encoder)
        z = y_hat_24k.new_zeros(())
        return LossBundle(spk=z, spec=z, f0=z, id=L_id, intel=z)

    L_id = speaker_cosine_loss_from_embed(e_prompt, y_hat_24k, spk_encoder)
    L_spk = speaker_cosine_loss_from_embed(e_prompt, y_hat_cycle_24k, spk_encoder)
    L_spec = mel_l1_loss(x_prompt_24k, y_hat_cycle_24k, mel_extractor)
    L_f0 = f0_pcc_loss(x_prompt_24k, y_hat_cycle_24k, f0_extractor)
    L_intel = y_hat_24k.new_zeros(())  # populated at eval-time (non-differentiable)
    return LossBundle(spk=L_spk, spec=L_spec, f0=L_f0, id=L_id, intel=L_intel)


# ---------------------------------------------------------------------------
# Outer loss
# ---------------------------------------------------------------------------

def outer_loss(
    x_prompt_24k: torch.Tensor,
    y_query_24k: torch.Tensor,
    *,
    spk_encoder: WavLMSpeakerEncoder,
    f0_extractor: F0Extractor,
    lambda_f0: float = 0.3,
) -> Tuple[torch.Tensor, dict]:
    """L_outer(τ) = (1 - SIM-o(x, ŷ_q)) + λ_f0 · (1 - F0_PCC(x, ŷ_q)).

    CER is omitted from the gradient because Whisper is autoregressive /
    non-differentiable; we report it post-hoc at evaluation time.
    """
    L_sim = speaker_cosine_loss(x_prompt_24k, y_query_24k, spk_encoder)
    L_f0 = f0_pcc_loss(x_prompt_24k, y_query_24k, f0_extractor)
    total = L_sim + lambda_f0 * L_f0
    return total, {"outer_sim": L_sim.detach(), "outer_f0": L_f0.detach()}
