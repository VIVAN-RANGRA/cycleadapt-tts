"""Frozen feature extractors used to compute cycle / forward losses and metrics.

All extractors expose a small functional interface that takes a waveform tensor
(at the natural sample rate of the underlying model, after our resampler) and
returns a per-utterance embedding or per-frame feature.  Everything that
participates in inner-loop gradients is differentiable; eval-only extractors
(Whisper, UTMOS, ECAPA-TDNN, Resemblyzer) are wrapped in :class:`torch.no_grad`.

Models loaded:
  * WavLM speaker verification (microsoft/wavlm-base-plus-sv) — primary SIM-o,
    used as ``L_spk`` and ``L_id`` during training.  Differentiable.
  * Vocos vocoder (re-uses the one already loaded by F5-TTS).
  * torchcrepe — pitch tracking for ``L_f0``.  Differentiable.
  * faster-whisper (large-v3) — WER/CER for evaluation.
  * Resemblyzer (GE2E) — secondary SECS for evaluation.
  * SpeechBrain ECAPA-TDNN — tertiary SECS, robustness check.
  * UTMOS — neural MOS predictor for evaluation.

The extractor objects are deliberately *lazy*: heavy models (Whisper, UTMOS,
Resemblyzer, ECAPA-TDNN) are not loaded until first use, so meta-training does
not pay their VRAM cost.
"""
from __future__ import annotations

import functools
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------------------------------------------------------------------------
# WavLM speaker verification (primary, differentiable)
# ---------------------------------------------------------------------------

class WavLMSpeakerEncoder(nn.Module):
    """Wraps microsoft/wavlm-base-plus-sv from HuggingFace transformers.

    Expects a 16 kHz mono waveform tensor of shape ``[B, T]``.  Returns an
    L2-normalised embedding of shape ``[B, 512]``.

    Stays in eval mode (no dropout); parameters are frozen.  ``forward`` is
    differentiable so we can backprop the SIM-o loss into the F5-TTS adapter.
    """

    def __init__(self, model_id: str = "microsoft/wavlm-base-plus-sv", device: str = "cuda"):
        super().__init__()
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        # HF feature extractor expects 16 kHz.
        self.target_sr = 16_000

    @torch.no_grad()
    def _normalize_waveform_for_fe(self, wav: torch.Tensor) -> torch.Tensor:
        """The HF feature extractor will normalize internally; we just need shape."""
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav

    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """wav_16k: [B, T] @ 16kHz, differentiable.  Returns [B, 512] L2-normalised."""
        # transformers' feature extractor uses numpy in the public API; here we
        # compute the equivalent normalised input directly in torch so gradients
        # flow.  WavLM uses per-utterance zero-mean unit-variance normalization.
        if wav_16k.ndim == 1:
            wav_16k = wav_16k.unsqueeze(0)
        x = wav_16k - wav_16k.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-8)
        out = self.model(input_values=x.to(self.device))
        emb = out.embeddings  # [B, 512]
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Both [..., D]; returns [..., ] cosine similarity."""
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return (a * b).sum(dim=-1)


# ---------------------------------------------------------------------------
# F0 (torchcrepe) — differentiable pitch
# ---------------------------------------------------------------------------

class F0Extractor(nn.Module):
    """Differentiable F0 extraction using torchcrepe (tiny model, fp16, 24kHz)."""

    def __init__(
        self,
        model: str = "tiny",
        hop_length_ms: float = 10.0,
        fmin: float = 50.0,
        fmax: float = 1100.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.model = model
        self.fmin = fmin
        self.fmax = fmax
        self.device = device
        self.target_sr = 16_000  # torchcrepe operates at 16kHz
        self.hop_length = int(self.target_sr * hop_length_ms / 1000)  # samples

    def forward(self, wav_16k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (f0 [B, N], voicing [B, N]) — both differentiable in wav_16k."""
        import torchcrepe

        if wav_16k.ndim == 1:
            wav_16k = wav_16k.unsqueeze(0)
        wav_16k = wav_16k.to(self.device)

        # Use weighted-argmax pitch decoding for differentiability.
        f0 = torchcrepe.predict(
            wav_16k,
            sample_rate=self.target_sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            model=self.model,
            decoder=torchcrepe.decode.weighted_argmax,
            device=self.device,
            return_periodicity=True,
            batch_size=2048,
            pad=True,
        )
        if isinstance(f0, tuple):
            f0_hz, periodicity = f0
        else:
            f0_hz = f0
            periodicity = torch.ones_like(f0_hz)
        return f0_hz, periodicity


# ---------------------------------------------------------------------------
# Mel spectrogram (24kHz, matches F5-TTS Vocos settings)
# ---------------------------------------------------------------------------

class MelSpecExtractor(nn.Module):
    """24 kHz, 100-channel mel spectrogram identical to F5-TTS Vocos config.

    We piggy-back on the ``cfm.mel_spec`` module already loaded with F5-TTS so
    train and eval use bit-identical spectrogram code.
    """

    def __init__(self, cfm_mel_spec: nn.Module):
        super().__init__()
        self.mel_spec = cfm_mel_spec
        # cfm.mel_spec has no nn.Parameters (only buffers); fall back to buffers.
        params = list(cfm_mel_spec.parameters())
        if params:
            self.device = params[0].device
        else:
            bufs = list(cfm_mel_spec.buffers())
            self.device = bufs[0].device if bufs else torch.device("cpu")

    def forward(self, wav_24k: torch.Tensor) -> torch.Tensor:
        """wav_24k: [B, T] -> mel: [B, 100, frames]."""
        if wav_24k.ndim == 1:
            wav_24k = wav_24k.unsqueeze(0)
        # cfm.mel_spec expects float32 wav input.
        return self.mel_spec(wav_24k.to(self.device).float())


# ---------------------------------------------------------------------------
# Whisper ASR (eval only)
# ---------------------------------------------------------------------------

class WhisperASR:
    """faster-whisper wrapper, eval-only.  Returns transcript string."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        from faster_whisper import WhisperModel
        cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=cache)

    def transcribe(self, wav: np.ndarray, sr: int, language: Optional[str] = None) -> str:
        if sr != 16000:
            wav = torchaudio.functional.resample(
                torch.as_tensor(wav, dtype=torch.float32), orig_freq=sr, new_freq=16000
            ).numpy()
        segments, _info = self.model.transcribe(
            wav,
            language=language,
            beam_size=1,
            without_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments)


# ---------------------------------------------------------------------------
# Resemblyzer (eval only)
# ---------------------------------------------------------------------------

class ResemblyzerEncoder:
    """GE2E speaker embedding, used for SECS metric."""

    def __init__(self, device: str = "cuda"):
        from resemblyzer import VoiceEncoder
        self.model = VoiceEncoder(device=torch.device(device), verbose=False)
        self.target_sr = 16_000

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(
                torch.as_tensor(wav, dtype=torch.float32),
                orig_freq=sr,
                new_freq=self.target_sr,
            ).numpy()
        return self.model.embed_utterance(wav)


# ---------------------------------------------------------------------------
# ECAPA-TDNN (eval only) — speechbrain
# ---------------------------------------------------------------------------

class ECAPATDNNEncoder(nn.Module):
    """speechbrain/spkrec-ecapa-voxceleb — tertiary speaker embedding."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        from speechbrain.inference.speaker import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path(os.environ.get("CYCLE_TTS_ROOT", "/home/ubuntu/CYCLE_TTS")) / "data/cache/sb_ecapa"),
            run_opts={"device": device},
        )
        self.target_sr = 16_000

    @torch.no_grad()
    def embed(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav.float(), orig_freq=sr, new_freq=self.target_sr)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        emb = self.model.encode_batch(wav)  # [B, 1, 192]
        return F.normalize(emb.squeeze(1), p=2, dim=-1)


# ---------------------------------------------------------------------------
# UTMOS (eval only)
# ---------------------------------------------------------------------------

class UTMOSPredictor:
    """UTMOS — neural MOS predictor.  Uses speechbrain's torch.hub mirror if available,
    falls back to ``sarulab-speech/UTMOS22`` from torch.hub.

    If UTMOS cannot be loaded (e.g. missing checkpoint on first run), the
    predictor returns ``nan`` so that downstream tables don't blow up.  The user
    can populate it later from a separate script.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        try:
            self.model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0",
                "utmos22_strong",
                trust_repo=True,
            ).to(device).eval()
            self.target_sr = 16_000
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"UTMOS unavailable, will return NaN: {e}")

    @torch.no_grad()
    def score(self, wav: torch.Tensor, sr: int) -> float:
        if self.model is None:
            return float("nan")
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav.float(), orig_freq=sr, new_freq=self.target_sr)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return float(self.model(wav.to(self.device), self.target_sr).mean().item())


# ---------------------------------------------------------------------------
# Resampler cache
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _resample_kernel(orig: int, new: int, device: str) -> torchaudio.transforms.Resample:
    return torchaudio.transforms.Resample(orig_freq=orig, new_freq=new).to(device)


def resample(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    """Differentiable resample using torchaudio."""
    if orig_sr == new_sr:
        return wav
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    device = wav.device.type
    kernel = _resample_kernel(orig_sr, new_sr, device)
    out = kernel(wav)
    if squeeze:
        out = out.squeeze(0)
    return out
