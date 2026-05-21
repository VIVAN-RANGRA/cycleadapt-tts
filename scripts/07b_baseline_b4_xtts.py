#!/usr/bin/env python
"""Baseline B4 — Coqui-TTS XTTS-v2 (multilingual zero-shot TTS).

XTTS-v2 supports en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja,
hu, ko, hi — covers all our eval L2s.

This script lazy-installs ``coqui-tts`` if missing, loads the model once, and
generates audio for every eval item that has a supported L2.

Output: ``results/audio/b4_xtts_v2/<pair_id>_<slot:03d>.wav``
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

ROOT = Path("/home/ubuntu/CYCLE_TTS")
sys.path.insert(0, str(ROOT / "src"))

from cycle_tts.eval_prompts import EvalItem, load_eval_set  # noqa: E402


XTTS_LANG = {
    "en": "en", "zh": "zh-cn", "es": "es", "fr": "fr", "de": "de",
    "hi": "hi", "ja": "ja", "ko": "ko",
}


def ensure_coqui_tts() -> None:
    try:
        import TTS  # noqa: F401
    except ImportError:
        print("[b4] installing coqui-tts ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "coqui-tts"])
        # Pull only the runtime extras we actually need.
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                                "encodec", "tokenizers", "trainer", "coqpit",
                                "num2words", "spacy", "gruut",
                                "cython", "tortoise-tts", "deepspeed"])


def load_prompt_wav(path: str, sr_target: int = 22_050) -> str:
    """XTTS reads the prompt from a file path; we re-write it as a clean .wav
    in /tmp at the desired SR to ensure compat (avoid mp3/flac edge cases)."""
    import tempfile
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wav.squeeze(0).numpy(), sr_target)
    return tmp.name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", default=str(ROOT / "results" / "eval_set.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "results" / "audio" / "b4_xtts_v2"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("b4")

    ensure_coqui_tts()
    os.environ["COQUI_TOS_AGREED"] = "1"

    from TTS.api import TTS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    timings_path = out_dir / "timings.jsonl"

    log.info("Loading XTTS-v2 …")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(args.device)

    items = load_eval_set(Path(args.eval_set))
    skipped = 0
    timings = []
    t_total = time.time()

    with timings_path.open("w") as tf:
        for i, it in enumerate(items):
            out_path = out_dir / f"{it.pair_id}_{it.slot:03d}.wav"
            if out_path.exists():
                continue
            xtts_lang = XTTS_LANG.get(it.L2)
            if xtts_lang is None:
                log.warning("XTTS does not support L2=%s; skipping %s", it.L2, out_path.name)
                skipped += 1
                continue
            t0 = time.time()
            try:
                prompt_path = load_prompt_wav(it.prompt_wav)
                tts.tts_to_file(
                    text=it.gen_text,
                    speaker_wav=prompt_path,
                    language=xtts_lang,
                    file_path=str(out_path),
                )
            except Exception as e:
                log.warning("Skipping %s: %s", out_path.name, e)
                skipped += 1
                continue
            dt = time.time() - t0
            # Re-load to read duration
            try:
                w, sr_o = torchaudio.load(str(out_path))
                gen_sec = w.shape[-1] / sr_o
            except Exception:
                gen_sec = float("nan")
            rec = {
                "item": out_path.name, "pair_id": it.pair_id,
                "L1": it.L1, "L2": it.L2,
                "gen_sec": float(gen_sec),
                "elapsed_sec": dt,
                "rtf": dt / max(gen_sec, 1e-6) if gen_sec else float("nan"),
            }
            tf.write(json.dumps(rec) + "\n")
            timings.append(rec)
            if (i + 1) % 25 == 0:
                log.info("[%d/%d]  %s  elapsed=%.0fs", i + 1, len(items), out_path.name, time.time() - t_total)

    t_total = time.time() - t_total
    valid_rtfs = [t["rtf"] for t in timings if t["rtf"] == t["rtf"]]
    summary = {
        "method": "b4_xtts_v2",
        "n_items": len(items),
        "n_skipped": skipped,
        "wall_seconds": t_total,
        "mean_rtf": (sum(valid_rtfs) / len(valid_rtfs)) if valid_rtfs else float("nan"),
        "out_dir": str(out_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("DONE  %d items skipped=%d  in %.0fs.  Summary -> %s",
             len(items) - skipped, skipped, t_total, summary_path)


if __name__ == "__main__":
    main()
