#!/usr/bin/env python
"""Baseline B5 — CosyVoice 2 (FunAudioLLM/CosyVoice).

CosyVoice 2 is a state-of-the-art autoregressive multilingual zero-shot TTS.
We use the official ``cosyvoice2-0.5B`` checkpoint from ModelScope/HuggingFace
via the official ``cosyvoice`` package (lazy-installed if missing).

Output: ``results/audio/b5_cosyvoice2/<pair_id>_<slot:03d>.wav``

Supported langs: zh, en, ja, ko, es, fr, de, ru, ar, … — covers everything
in our eval set.

This script is **best-effort**: if the CosyVoice install or model fetch
fails on this machine we skip B5 gracefully (writes an empty
``summary.json`` so downstream eval scripts know to ignore it).
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

from cycle_tts.eval_prompts import load_eval_set  # noqa: E402


COSYVOICE_REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"


def ensure_cosyvoice() -> bool:
    """Lazy-install the CosyVoice runtime.  Returns True iff the import works."""
    try:
        import cosyvoice  # noqa: F401
        return True
    except ImportError:
        pass
    print("[b5] attempting to install cosyvoice ...", flush=True)
    try:
        # The official package is published on PyPI as ``cosyvoice`` (newer
        # releases) — try pip first; if that fails fall back to a git
        # snapshot.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cosyvoice"])
        import cosyvoice  # noqa: F401
        return True
    except Exception as e:
        print(f"[b5] pip install cosyvoice failed: {e}", flush=True)
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", default=str(ROOT / "results" / "eval_set.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "results" / "audio" / "b5_cosyvoice2"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("b5")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    timings_path = out_dir / "timings.jsonl"

    if not ensure_cosyvoice():
        log.warning("Could not install cosyvoice; skipping B5.")
        summary_path.write_text(json.dumps({
            "method": "b5_cosyvoice2", "status": "skipped",
            "reason": "cosyvoice package not installable in this environment",
        }, indent=2))
        return

    from cosyvoice.cli.cosyvoice import CosyVoice2

    log.info("Loading CosyVoice 2 (this can download ~2GB on first run) …")
    try:
        cosy = CosyVoice2(COSYVOICE_REPO_ID)
    except Exception as e:
        log.warning("CosyVoice2 load failed: %s — skipping B5.", e)
        summary_path.write_text(json.dumps({
            "method": "b5_cosyvoice2", "status": "skipped", "reason": str(e),
        }, indent=2))
        return

    items = load_eval_set(Path(args.eval_set))
    skipped = 0
    timings = []
    t_total = time.time()

    with timings_path.open("w") as tf:
        for i, it in enumerate(items):
            out_path = out_dir / f"{it.pair_id}_{it.slot:03d}.wav"
            if out_path.exists():
                continue
            t0 = time.time()
            try:
                prompt_wav, prompt_sr = torchaudio.load(it.prompt_wav)
                if prompt_wav.shape[0] > 1:
                    prompt_wav = prompt_wav.mean(0, keepdim=True)
                if prompt_sr != 16_000:
                    prompt_wav = torchaudio.functional.resample(prompt_wav, prompt_sr, 16_000)
                pieces = []
                for piece in cosy.inference_zero_shot(
                    it.gen_text,
                    it.prompt_text,
                    prompt_wav,
                    stream=False,
                ):
                    pieces.append(piece["tts_speech"])
                wave = torch.cat(pieces, dim=-1).squeeze().float().cpu().numpy()
                sf.write(out_path, wave, 24_000)
            except Exception as e:
                log.warning("Skipping %s: %s", out_path.name, e)
                skipped += 1
                continue
            dt = time.time() - t0
            gen_sec = float(wave.shape[-1] / 24_000)
            rec = {
                "item": out_path.name, "pair_id": it.pair_id,
                "L1": it.L1, "L2": it.L2,
                "gen_sec": gen_sec, "elapsed_sec": dt,
                "rtf": dt / max(gen_sec, 1e-6),
            }
            tf.write(json.dumps(rec) + "\n")
            timings.append(rec)
            if (i + 1) % 25 == 0:
                log.info("[%d/%d]  %s  elapsed=%.0fs", i + 1, len(items), out_path.name, time.time() - t_total)

    t_total = time.time() - t_total
    valid_rtfs = [t["rtf"] for t in timings if t["rtf"] == t["rtf"]]
    summary_path.write_text(json.dumps({
        "method": "b5_cosyvoice2",
        "n_items": len(items),
        "n_skipped": skipped,
        "wall_seconds": t_total,
        "mean_rtf": (sum(valid_rtfs) / len(valid_rtfs)) if valid_rtfs else float("nan"),
        "out_dir": str(out_dir),
    }, indent=2))
    log.info("DONE  %d items skipped=%d  in %.0fs.", len(items) - skipped, skipped, t_total)


if __name__ == "__main__":
    main()
