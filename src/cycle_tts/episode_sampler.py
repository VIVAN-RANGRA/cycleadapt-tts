"""Cross-lingual episode sampler.

Each episode is a tuple ``(speaker_s, L1, L2, x, t, t_q, t_prime)`` where:

  * ``L1``: prompt language
  * ``L2``: target language (L2 ≠ L1)
  * ``x``: speaker prompt waveform @ 24 kHz in L1
  * ``t``: support text in L2
  * ``t_q``: query text in L2 (≠ t)
  * ``t_prime``: source-language transcript of ``x``

Sampling strategy
-----------------
1. Pick a (L1, L2) pair according to ``cfg.train_lang_pairs`` weights.
2. Pick a random speaker from L1's manifest.
3. Pick a random utterance for that speaker — that gives ``x`` and ``t_prime``.
4. Pick ``t`` and ``t_q`` from the L2 sentence pool (distinct).

We bias prompt selection toward 3–7 s audio (cfg.data.prompt_min_sec /
prompt_max_sec) to keep meta-training fast.  If a sampled utterance falls
outside the window, we crop or skip it (with a max retry of 8).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch

from .config import CycleAdaptConfig
from .data import ManifestIndex, Utterance, load_audio_24k
from .text_pool import sample_pair


@dataclass
class Episode:
    speaker_id: str
    lang_l1: str
    lang_l2: str
    lang_l1_idx: int
    lang_l2_idx: int
    x_wav_24k: torch.Tensor   # [1, T]
    t_support: str            # target text in L2 for inner-loop support
    t_query: str              # target text in L2 for outer-loop evaluation
    t_prime: str              # transcript of x (in L1)


class EpisodeSampler:
    def __init__(self, cfg: CycleAdaptConfig, manifest: ManifestIndex, *, seed: int = 0):
        self.cfg = cfg
        self.manifest = manifest
        self.rng = random.Random(seed)

        # Filter the configured lang pairs to those for which we have data.
        usable = []
        for L1, L2, w in cfg.train_lang_pairs:
            if L1 in manifest.by_lang_speaker and manifest.n_speakers(L1) > 0:
                usable.append((L1, L2, w))
        if not usable:
            raise RuntimeError("No usable language pairs — manifest has no L1 speakers.")
        self.pairs = usable

    def _pick_pair(self) -> tuple[str, str]:
        weights = [w for _, _, w in self.pairs]
        L1, L2, _ = self.rng.choices(self.pairs, weights=weights, k=1)[0]
        return L1, L2

    def _crop_or_pad_wav(self, wav: torch.Tensor) -> torch.Tensor:
        """Crop a long prompt to [prompt_min, prompt_max] seconds."""
        sr = 24_000
        cfg = self.cfg.data
        min_samples = int(cfg.prompt_min_sec * sr)
        max_samples = int(cfg.prompt_max_sec * sr)
        T = wav.shape[-1]
        if T < min_samples:
            # Too short — duplicate to reach minimum (handles short LibriTTS-R clips).
            n_rep = (min_samples + T - 1) // T
            wav = wav.repeat(1, n_rep)[:, :min_samples]
            T = wav.shape[-1]
        if T > max_samples:
            start = self.rng.randint(0, T - max_samples)
            wav = wav[:, start:start + max_samples]
        return wav

    def sample(self) -> Optional[Episode]:
        for _ in range(8):
            L1, L2 = self._pick_pair()
            try:
                speaker_id = self.manifest.random_speaker(L1, self.rng)
                row = self.manifest.random_utterance_for(L1, speaker_id, self.rng)
                wav = load_audio_24k(row)
                wav = self._crop_or_pad_wav(wav)
            except Exception:
                continue
            if wav is None or wav.numel() < 24_000:  # skip <1s
                continue

            t_support, t_query = sample_pair(L2, self.rng)
            return Episode(
                speaker_id=speaker_id,
                lang_l1=L1,
                lang_l2=L2,
                lang_l1_idx=self.cfg.lang_idx[L1],
                lang_l2_idx=self.cfg.lang_idx[L2],
                x_wav_24k=wav,
                t_support=t_support,
                t_query=t_query,
                t_prime=row["text"],
            )
        return None
