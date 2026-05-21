"""Manifest builder + per-utterance dataset for CycleAdapt-TTS.

A "manifest" is a JSONL where each row is one utterance:

    {
      "source": "libritts_r" | "aishell3" | "vctk" | "fleurs_es" | ...,
      "speaker_id": "LibriTTS_R_103" | "aishell3_SSB0005" | "vctk_p225" | ...,
      "lang": "en" | "zh" | "es",
      "split": "train" | "eval",
      "shard": <path>,           # parquet shard path, or absolute wav/flac path for VCTK
      "row_or_path": <int|str>,  # row index (parquet) or filename (vctk webdataset)
      "text": "...",
      "duration_sec": 5.32,
      "sample_rate": 24000,      # natural sr (will be resampled to 24kHz on-the-fly)
    }

We build separate manifests for *train* (meta-learning) and *eval* (held-out
speakers).  VCTK is entirely held out; Common Voice / FLEURS are held out as
eval sets per the plan §3.2.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio


CYCLE_ROOT = Path(os.environ.get("CYCLE_TTS_ROOT", "/home/ubuntu/CYCLE_TTS"))
DATA = CYCLE_ROOT / "data"


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def _safe_load_parquet(path: Path):
    return pq.read_table(path, memory_map=True)


def _build_libritts_r_manifest(out_path: Path) -> int:
    """LibriTTS-R train.clean.100 — 247 EN speakers, ~33h, 24kHz."""
    shards_dir = DATA / "libritts_r" / "data" / "train.clean.100"
    shards = sorted(shards_dir.glob("*.parquet"))
    n = 0
    with out_path.open("w") as fout:
        for sh in shards:
            table = _safe_load_parquet(sh)
            cols = table.column_names
            assert "audio" in cols and "text_normalized" in cols, f"unexpected schema: {cols}"
            audio_arr = table.column("audio")
            text_arr = table.column("text_normalized")
            spk_col = "speaker_id" if "speaker_id" in cols else None
            for i in range(len(table)):
                audio_d = audio_arr[i].as_py()
                if audio_d is None:
                    continue
                audio_path = audio_d.get("path") or ""
                m = re.match(r"^(\d+)_", os.path.basename(audio_path))
                speaker_id = (table[spk_col][i].as_py() if spk_col else None) or (m.group(1) if m else "unk")
                text = text_arr[i].as_py() or ""
                # Approx duration from byte count is wrong; we'll back-compute via soundfile lazily.
                fout.write(json.dumps({
                    "source": "libritts_r",
                    "speaker_id": f"libritts_r_{speaker_id}",
                    "lang": "en",
                    "split": "train",
                    "shard": str(sh),
                    "row_or_path": i,
                    "text": text,
                    "duration_sec": None,
                    "sample_rate": 24_000,
                }) + "\n")
                n += 1
    return n


def _build_aishell3_manifest(out_path: Path) -> int:
    """AISHELL-3 — 218 ZH speakers, 44.1kHz, parquet with (audio, text, pinyin) columns."""
    shards = sorted((DATA / "aishell3_raw" / "data").glob("train-*.parquet"))
    n = 0
    with out_path.open("w") as fout:
        for sh in shards:
            table = _safe_load_parquet(sh)
            cols = table.column_names
            audio_arr = table.column("audio")
            text_arr = table.column("text")
            for i in range(len(table)):
                audio_d = audio_arr[i].as_py()
                if audio_d is None:
                    continue
                audio_path = audio_d.get("path") or ""
                # AISHELL-3 paths look like "SSB06930002.wav" — speaker = first 7 chars.
                m = re.match(r"^(SSB\d{4})", os.path.basename(audio_path))
                speaker_id = m.group(1) if m else "unk"
                text = text_arr[i].as_py() or ""
                fout.write(json.dumps({
                    "source": "aishell3",
                    "speaker_id": f"aishell3_{speaker_id}",
                    "lang": "zh",
                    "split": "train",
                    "shard": str(sh),
                    "row_or_path": i,
                    "text": text,
                    "duration_sec": None,
                    "sample_rate": 44_100,  # AISHELL-3 native; will be resampled
                }) + "\n")
                n += 1
    return n


def _build_vctk_manifest(out_path: Path, eval_split: bool = True) -> int:
    """VCTK extracted into webdataset-style files in ``data/vctk_extracted/``.

    Each utterance has paired ``{id}.flac`` + ``{id}.normalized.txt`` + ``{id}.speaker_id``.
    All utterances are marked as **eval** (held-out per plan §3.2).
    """
    root = DATA / "vctk_extracted"
    if not root.exists():
        return 0
    flacs = sorted(root.rglob("*.flac"))
    n = 0
    with out_path.open("w") as fout:
        for flac in flacs:
            stem = flac.with_suffix("")
            try:
                text = (stem.with_suffix(".normalized.txt")).read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                try:
                    text = (stem.with_suffix(".txt")).read_text(encoding="utf-8").strip()
                except FileNotFoundError:
                    continue
            try:
                speaker_id = (stem.with_suffix(".speaker_id")).read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                speaker_id = "unk"
            fout.write(json.dumps({
                "source": "vctk",
                "speaker_id": f"vctk_{speaker_id}",
                "lang": "en",
                "split": "eval" if eval_split else "train",
                "shard": str(flac),
                "row_or_path": flac.name,
                "text": text,
                "duration_sec": None,
                "sample_rate": 48_000,
            }) + "\n")
            n += 1
    return n


def _build_fleurs_manifest(out_path: Path) -> int:
    """FLEURS test split for es_419 and cmn_hans_cn — held-out eval prompts."""
    root = DATA / "fleurs_extracted"
    if not root.exists():
        return 0
    rows = []
    for lang_dir in root.iterdir():
        if not lang_dir.is_dir():
            continue
        lang_code = lang_dir.name
        # Map FLEURS BCP-47 codes to our ISO-2 codes.
        lang = {
            "es_419": "es",
            "cmn_hans_cn": "zh",
            "en_us": "en",
            "fr_fr": "fr",
            "de_de": "de",
            "hi_in": "hi",
            "ja_jp": "ja",
            "ko_kr": "ko",
            "it_it": "it",
            "pt_br": "pt",
            "ru_ru": "ru",
        }.get(lang_code)
        if lang is None:
            continue
        tsv = DATA / "fleurs" / "data" / lang_code / "test.tsv"
        wav_dir = lang_dir / "test"
        if not tsv.exists() or not wav_dir.exists():
            continue
        # FLEURS tsv columns: id, filename, raw_transcription, transcription, num_samples, gender
        for line in tsv.read_text().strip().split("\n"):
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            fname = parts[1]
            raw_text = parts[2]
            speaker_id = f"fleurs_{lang}_{parts[0]}"  # unique per-utterance id (FLEURS has no speaker labels)
            wav_path = wav_dir / fname
            if not wav_path.exists():
                continue
            rows.append({
                "source": f"fleurs_{lang}",
                "speaker_id": speaker_id,
                "lang": lang,
                "split": "eval",
                "shard": str(wav_path),
                "row_or_path": fname,
                "text": raw_text,
                "duration_sec": None,
                "sample_rate": 16_000,
            })
    with out_path.open("w") as fout:
        for r in rows:
            fout.write(json.dumps(r) + "\n")
    return len(rows)


def build_all_manifests(out_dir: Optional[Path] = None) -> Dict[str, int]:
    """Build train and eval manifests, returning counts."""
    out_dir = Path(out_dir or DATA / "manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    counts["libritts_r_train"] = _build_libritts_r_manifest(out_dir / "libritts_r_train.jsonl")
    counts["aishell3_train"] = _build_aishell3_manifest(out_dir / "aishell3_train.jsonl")
    counts["vctk_eval"] = _build_vctk_manifest(out_dir / "vctk_eval.jsonl")
    counts["fleurs_eval"] = _build_fleurs_manifest(out_dir / "fleurs_eval.jsonl")
    return counts


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    speaker_id: str
    lang: str
    text: str
    wav_24k: torch.Tensor  # [1, T]
    sample_rate: int = 24_000


_PARQUET_CACHE: Dict[str, Any] = {}


def _open_parquet(path: str):
    if path not in _PARQUET_CACHE:
        _PARQUET_CACHE[path] = _safe_load_parquet(Path(path))
    return _PARQUET_CACHE[path]


def load_audio_24k(row: Dict[str, Any]) -> torch.Tensor:
    """Load and resample to 24kHz mono."""
    src = row["source"]
    if src in ("libritts_r", "aishell3") or src.startswith("aishell3") or src.startswith("libritts"):
        table = _open_parquet(row["shard"])
        audio_d = table.column("audio")[row["row_or_path"]].as_py()
        wav_bytes = audio_d["bytes"]
        wav, sr = sf.read(BytesIO(wav_bytes), dtype="float32")
    else:
        wav, sr = sf.read(row["shard"], dtype="float32")
    if wav.ndim == 2:  # stereo
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
    if sr != 24_000:
        wav = torchaudio.functional.resample(wav, sr, 24_000)
    return wav


# ---------------------------------------------------------------------------
# Manifest in-memory index
# ---------------------------------------------------------------------------

@dataclass
class ManifestIndex:
    """In-memory index over a JSONL manifest grouped by (lang, speaker)."""

    rows: List[Dict[str, Any]]
    by_lang_speaker: Dict[str, Dict[str, List[int]]]  # lang -> speaker -> [row_idx]
    by_lang: Dict[str, List[int]]                     # lang -> [row_idx]

    @classmethod
    def from_jsonl_files(cls, paths: Sequence[Path]) -> "ManifestIndex":
        rows: List[Dict[str, Any]] = []
        for p in paths:
            if not Path(p).exists():
                continue
            for line in Path(p).read_text().split("\n"):
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        by_lang_speaker: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        by_lang: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            by_lang_speaker[r["lang"]][r["speaker_id"]].append(i)
            by_lang[r["lang"]].append(i)
        return cls(rows=rows, by_lang_speaker=dict(by_lang_speaker), by_lang=dict(by_lang))

    def n_speakers(self, lang: str) -> int:
        return len(self.by_lang_speaker.get(lang, {}))

    def random_speaker(self, lang: str, rng: random.Random) -> str:
        return rng.choice(list(self.by_lang_speaker[lang].keys()))

    def random_utterance_for(self, lang: str, speaker_id: str, rng: random.Random) -> Dict[str, Any]:
        idx = rng.choice(self.by_lang_speaker[lang][speaker_id])
        return self.rows[idx]
