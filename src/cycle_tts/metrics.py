"""Objective evaluation metrics for CycleAdapt-TTS.

All metrics operate on lists of ``EvalRecord``::

    EvalRecord(item_id, prompt_wav, gen_wav, prompt_text, gen_text, L1, L2)

where ``prompt_wav`` is the *original prompt* (24 kHz mono tensor) and
``gen_wav`` is the *generated audio* (24 kHz mono tensor) produced by some
method (baseline or ours).

Metrics
-------
* **SIM-o (WavLM)**: cosine sim between WavLM-base-plus-sv embeddings.
* **SIM-o (ECAPA)**: cosine sim between SpeechBrain ECAPA-TDNN embeddings.
* **F0-PCC**: Pearson correlation of log-F0 contours (jointly voiced frames).
* **ASR error**: faster-whisper edit error vs. ``gen_text``; WER for
  whitespace-tokenized languages and CER for CJK.
* **UTMOS**: predicted MOS from speechmos UTMOS22.

Lazy-loading
------------
Each metric loads its model on first call.  This is important because we
have multiple methods to evaluate; computing all metrics for one method only
incurs a one-time model load cost.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    item_id: str            # unique key (typically "<pair_id>_<slot>")
    pair_id: str            # "{L1}_{L2}"
    pair_class: str         # "in-distrib" / "zero-shot"
    L1: str
    L2: str
    prompt_wav: torch.Tensor  # [T] @ 24 kHz, float32
    gen_wav: torch.Tensor     # [T] @ 24 kHz, float32
    prompt_text: str
    gen_text: str
    method: str


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------

def _resample(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wav
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    out = torchaudio.functional.resample(wav, sr_in, sr_out)
    return out.squeeze(0)


def _to_mono_16k(wav_24k: torch.Tensor) -> torch.Tensor:
    if wav_24k.ndim == 2:
        wav_24k = wav_24k.mean(0)
    return _resample(wav_24k, 24_000, 16_000)


# ---------------------------------------------------------------------------
# Metric: SIM-o (WavLM)
# ---------------------------------------------------------------------------

class WavLMSim:
    def __init__(self, device: str = "cuda"):
        from transformers import WavLMForXVector
        self.device = device
        self.model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed(self, wav_24k: torch.Tensor) -> torch.Tensor:
        x = _to_mono_16k(wav_24k.float()).unsqueeze(0).to(self.device)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        emb = self.model(input_values=x).embeddings  # [1, 512]
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.squeeze(0).cpu()

    def score(self, rec: EvalRecord) -> float:
        e1 = self.embed(rec.prompt_wav)
        e2 = self.embed(rec.gen_wav)
        return float(F.cosine_similarity(e1, e2, dim=-1).item())


# ---------------------------------------------------------------------------
# Metric: SIM-o (ECAPA via SpeechBrain)
# ---------------------------------------------------------------------------

class ECAPASim:
    def __init__(self, device: str = "cuda"):
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/home/ubuntu/CYCLE_TTS/data/cache/speechbrain_ecapa",
            run_opts={"device": device},
        )

    @torch.no_grad()
    def embed(self, wav_24k: torch.Tensor) -> torch.Tensor:
        x = _to_mono_16k(wav_24k.float()).unsqueeze(0).to(self.device)
        emb = self.model.encode_batch(x).squeeze(0).squeeze(0).cpu()
        return F.normalize(emb, p=2, dim=-1)

    def score(self, rec: EvalRecord) -> float:
        e1 = self.embed(rec.prompt_wav)
        e2 = self.embed(rec.gen_wav)
        return float(F.cosine_similarity(e1, e2, dim=-1).item())


# ---------------------------------------------------------------------------
# Metric: F0 PCC
# ---------------------------------------------------------------------------

class F0PCC:
    def __init__(self, device: str = "cuda"):
        import torchcrepe
        self.torchcrepe = torchcrepe
        self.device = device
        self.target_sr = 16_000
        self.hop_length = int(self.target_sr * 0.01)  # 10 ms

    @torch.no_grad()
    def _pitch(self, wav_24k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _to_mono_16k(wav_24k.float()).unsqueeze(0).to(self.device)
        try:
            f0, peri = self.torchcrepe.predict(
                x,
                self.target_sr,
                self.hop_length,
                50, 550,
                "tiny",
                decoder=self.torchcrepe.decode.weighted_argmax,
                device=self.device,
                return_periodicity=True,
                batch_size=2048,
            )
        except Exception:
            return x.new_zeros(1, 16), x.new_zeros(1, 16)
        return f0.squeeze(0), peri.squeeze(0)

    def score(self, rec: EvalRecord) -> float:
        f0a, pa = self._pitch(rec.prompt_wav)
        f0b, pb = self._pitch(rec.gen_wav)
        n = min(f0a.shape[-1], f0b.shape[-1])
        if n < 8:
            return float("nan")
        f0a, f0b, pa, pb = f0a[:n], f0b[:n], pa[:n], pb[:n]
        voiced = (pa > 0.5) & (pb > 0.5)
        if voiced.sum() < 8:
            return float("nan")
        la = torch.log(f0a.clamp(min=1.0))[voiced]
        lb = torch.log(f0b.clamp(min=1.0))[voiced]
        la = la - la.mean()
        lb = lb - lb.mean()
        num = (la * lb).sum()
        den = la.pow(2).sum().sqrt() * lb.pow(2).sum().sqrt()
        if den < 1e-8:
            return float("nan")
        return float((num / den).item())


# ---------------------------------------------------------------------------
# Metric: WER via faster-whisper
# ---------------------------------------------------------------------------

# A very small text normalization for cross-language WER. We DO NOT use
# whisper_normalizer here because it is English-centric; instead we lower-case,
# strip punctuation, and collapse whitespace.  This is what most multilingual
# TTS papers actually report.

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _norm_text(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _edit_distance(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


class WhisperWER:
    """faster-whisper ASR error.  Loads ``large-v3-turbo`` once.

    The returned metric is WER for whitespace-tokenized languages and CER for
    CJK languages.  The historical JSON key is still ``wer`` for compatibility.
    """

    LANG_TO_WHISPER = {
        "en": "en", "zh": "zh", "es": "es", "fr": "fr",
        "de": "de", "hi": "hi", "ja": "ja", "ko": "ko",
    }

    def __init__(
        self,
        device: str = "cuda",
        model_size: str = "large-v3-turbo",
        compute_type: str | None = None,
    ):
        from faster_whisper import WhisperModel
        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, wav_24k: torch.Tensor, lang: str) -> str:
        w = _to_mono_16k(wav_24k.float()).numpy()
        wlang = self.LANG_TO_WHISPER.get(lang)
        segments, _ = self.model.transcribe(
            w,
            language=wlang,
            task="transcribe",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = " ".join(s.text for s in segments)
        return text

    def score(self, rec: EvalRecord) -> float:
        hyp = self.transcribe(rec.gen_wav, rec.L2)
        # CJK languages: use character-level error rate (CER) rather than WER.
        ref_words = list(_norm_text(rec.gen_text)) if rec.L2 in ("zh", "ja", "ko") else _norm_text(rec.gen_text).split()
        hyp_words = list(_norm_text(hyp))        if rec.L2 in ("zh", "ja", "ko") else _norm_text(hyp).split()
        if not ref_words:
            return float("nan")
        d = _edit_distance(ref_words, hyp_words)
        return d / len(ref_words)


# ---------------------------------------------------------------------------
# Metric: UTMOS (predicted MOS)
# ---------------------------------------------------------------------------

class UTMOS:
    """Predicted MOS.

    We try (in order):
      * UTMOS (``speechmos.utmos22_strong``) if available,
      * DNSMOS (``speechmos.dnsmos``) — the OVRL (overall) score,

    which both produce a single scalar in [1, 5] interpretable as a "MOS"
    estimate.  We report whichever is loaded in the ``backend`` field of the
    summary.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.backend = None
        self._fn = None
        try:
            from speechmos import utmos22_strong  # type: ignore
            self._fn = utmos22_strong.run
            self.backend = "utmos22"
            return
        except ImportError:
            pass
        try:
            from speechmos import dnsmos  # type: ignore
            self._fn = dnsmos.run
            self.backend = "dnsmos_ovrl"
            return
        except ImportError:
            log.warning("Neither speechmos.utmos22_strong nor dnsmos available; UTMOS will return NaN")

    def score(self, rec: EvalRecord) -> float:
        if self._fn is None:
            return float("nan")
        w16 = _to_mono_16k(rec.gen_wav.float()).numpy()
        try:
            out = self._fn(w16, 16_000)
        except TypeError:
            out = self._fn({"audio": w16, "sample_rate": 16_000})
        if self.backend == "utmos22":
            return float(out["utmos"])
        return float(out["ovrl_mos"])
