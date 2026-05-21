#!/usr/bin/env python
"""Baseline B1 — vanilla F5-TTS (no adapter, no TTT).

Runs ``F5CycleWrapper.generate_eval`` on each item of the eval set and writes
the generated wave to::

    results/audio/{run_name}/{pair_id}_{slot:03d}.wav

Run-time: ~3 s/item * 250 items = ~13 minutes on this A100.  Safe to launch
in parallel with main meta-training; uses ~5 GB GPU memory.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

ROOT = Path("/home/ubuntu/CYCLE_TTS")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party_f5" / "src"))

from cycle_tts.eval_prompts import load_eval_set  # noqa: E402
from cycle_tts.f5_wrapper import F5CycleWrapper, load_f5tts_model  # noqa: E402


def load_prompt_wav_24k(path: str, src_sr: int = None) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 24_000:
        wav = torchaudio.functional.resample(wav, sr, 24_000)
    return wav.squeeze(0)  # [T]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", type=str, default=str(ROOT / "results" / "eval_set.jsonl"))
    p.add_argument("--out-dir", type=str, default=str(ROOT / "results" / "audio" / "b1_f5_vanilla"))
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--cfg-strength", type=float, default=2.0)
    p.add_argument("--sampler", choices=["diff", "eval"], default="diff",
                   help="diff matches the adaptive-method generation path; eval keeps the legacy no-adapter path")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("b1")

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings_path = out_dir / "timings.jsonl"
    summary_path = out_dir / "summary.json"

    log.info("Loading F5-TTS …")
    cfm, vocoder, vocab_char_map = load_f5tts_model(device=args.device, bf16=True)
    f5 = F5CycleWrapper(cfm, vocoder, vocab_char_map)

    items = load_eval_set(Path(args.eval_set))
    log.info("Eval items: %d", len(items))

    t_total = time.time()
    timings = []
    skipped = 0
    existing_skipped = 0

    with timings_path.open("w") as tf:
        for i, it in enumerate(items):
            out_path = out_dir / f"{it.pair_id}_{it.slot:03d}.wav"
            if out_path.exists() and not args.overwrite:
                existing_skipped += 1
                continue
            if out_path.exists() and args.overwrite:
                out_path.unlink()
            try:
                prompt_wav = load_prompt_wav_24k(it.prompt_wav)
            except Exception as e:
                log.warning("Skipping %s (prompt load failed): %s", out_path.name, e)
                skipped += 1
                continue

            t0 = time.time()
            try:
                with torch.inference_mode():
                    if args.sampler == "eval":
                        out = f5.generate_eval(
                            prompt_wav,
                            it.prompt_text,
                            it.gen_text,
                            nfe_step=args.nfe,
                            cfg_strength=args.cfg_strength,
                            seed=42,
                        )
                    else:
                        out = f5.generate_diff(
                            prompt_wav,
                            it.prompt_text,
                            it.gen_text,
                            nfe_step=args.nfe,
                            cfg_strength=args.cfg_strength,
                            sway_sampling_coef=-1.0,
                            overrides=None,
                            seed=42,
                        )
                wave = out.wave.squeeze().float().cpu().numpy()
                sf.write(out_path, wave, 24_000)
            except Exception as e:
                log.warning("Skipping %s (gen failed): %s", out_path.name, e)
                skipped += 1
                continue
            dt = time.time() - t0
            rec = {
                "item": out_path.name,
                "pair_id": it.pair_id,
                "L1": it.L1, "L2": it.L2,
                "speaker": it.speaker_id,
                "gen_sec": float(wave.shape[-1] / 24_000),
                "elapsed_sec": dt,
                "rtf": dt / max(wave.shape[-1] / 24_000, 1e-6),
            }
            tf.write(json.dumps(rec) + "\n")
            timings.append(rec)
            if (i + 1) % 25 == 0:
                log.info("[%d/%d]  %s  RTF=%.2f  elapsed=%.0fs",
                         i + 1, len(items), out_path.name, rec["rtf"], time.time() - t_total)

    t_total = time.time() - t_total
    summary = {
        "method": "b1_f5_vanilla",
        "n_items": len(items),
        "n_generated": len(timings),
        "n_existing_skipped": existing_skipped,
        "n_skipped": skipped,
        "final_nfe": args.nfe,
        "final_cfg_strength": args.cfg_strength,
        "sampler": args.sampler,
        "wall_seconds": t_total,
        "mean_rtf": (sum(t["rtf"] for t in timings) / len(timings)) if timings else float("nan"),
        "out_dir": str(out_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("DONE  %d items in %.0fs (%.2f s/item).  Summary -> %s",
             len(items) - skipped, t_total, t_total / max(len(items) - skipped, 1), summary_path)


if __name__ == "__main__":
    main()
