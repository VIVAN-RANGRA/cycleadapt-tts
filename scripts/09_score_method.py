#!/usr/bin/env python
"""Score a method's generated audio against the eval set.

Usage::

    python scripts/09_score_method.py \
        --gen-dir results/audio/b1_f5_vanilla \
        --eval-set results/eval_set.jsonl \
        --method b1_f5_vanilla \
        --out results/scores/b1_f5_vanilla.jsonl \
        --metrics simwavlm simecapa f0pcc wer utmos

Each metric is loaded lazily.  Output is per-item JSONL + a pair-level
``summary.json`` containing mean / std / 95% CI per metric per (L1, L2).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio

ROOT = Path("/home/ubuntu/CYCLE_TTS")
sys.path.insert(0, str(ROOT / "src"))

from cycle_tts.eval_prompts import load_eval_set  # noqa: E402
from cycle_tts.metrics import (  # noqa: E402
    ECAPASim, EvalRecord, F0PCC, UTMOS, WavLMSim, WhisperWER,
)


def load_wav(path: str, target_sr: int = 24_000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05) -> Dict[str, float]:
    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))])
    if len(arr) < 2:
        return {"mean": float(arr.mean()) if len(arr) else float("nan"),
                "std": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n": int(len(arr))}
    rng = np.random.default_rng(0)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci_lo": float(np.quantile(samples, alpha / 2)),
        "ci_hi": float(np.quantile(samples, 1 - alpha / 2)),
        "n": int(len(arr)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dir", required=True, help="dir of <pair_id>_<slot:03d>.wav")
    p.add_argument("--eval-set", default=str(ROOT / "results" / "eval_set.jsonl"))
    p.add_argument("--method", required=True)
    p.add_argument("--out", required=True, help="per-item JSONL output path")
    p.add_argument("--metrics", nargs="+", default=["simwavlm", "simecapa", "f0pcc", "wer", "utmos"],
                   choices=["simwavlm", "simecapa", "f0pcc", "wer", "utmos"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-missing", action="store_true",
                   help="score a partial generation directory instead of failing on missing wavs")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("score")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")

    gen_dir = Path(args.gen_dir)
    items = load_eval_set(Path(args.eval_set))
    gen_summary_path = gen_dir / "summary.json"
    gen_summary = {}
    if gen_summary_path.exists():
        try:
            gen_summary = json.loads(gen_summary_path.read_text())
        except Exception as e:
            log.warning("Could not read generation summary %s: %s", gen_summary_path, e)

    # Resolve which items have a generated wav file present.
    available = []
    for it in items:
        fn = gen_dir / f"{it.pair_id}_{it.slot:03d}.wav"
        if fn.exists():
            available.append((it, fn))
    log.info("Found %d/%d generated wavs in %s", len(available), len(items), gen_dir)
    if len(available) != len(items) and not args.allow_missing:
        raise SystemExit(
            f"Only found {len(available)}/{len(items)} generated wavs in {gen_dir}; "
            "use --allow-missing only for debugging."
        )

    # Lazy-load metric models (only requested ones).
    metric_objs = {}
    if "simwavlm" in args.metrics: metric_objs["simwavlm"] = WavLMSim(args.device)
    if "simecapa" in args.metrics: metric_objs["simecapa"] = ECAPASim(args.device)
    if "f0pcc" in args.metrics: metric_objs["f0pcc"] = F0PCC(args.device)
    if "wer" in args.metrics: metric_objs["wer"] = WhisperWER(args.device)
    if "utmos" in args.metrics: metric_objs["utmos"] = UTMOS(args.device)
    log.info("Loaded metrics: %s", list(metric_objs.keys()))

    per_item_rows: List[Dict] = []
    t0 = time.time()
    with out_path.open("w") as fout:
        for i, (it, fn) in enumerate(available):
            try:
                prompt_wav = load_wav(it.prompt_wav)
                gen_wav = load_wav(str(fn))
            except Exception as e:
                log.warning("Skipping %s: %s", fn.name, e)
                continue
            rec = EvalRecord(
                item_id=f"{it.pair_id}_{it.slot:03d}",
                pair_id=it.pair_id, pair_class=it.pair_class,
                L1=it.L1, L2=it.L2,
                prompt_wav=prompt_wav, gen_wav=gen_wav,
                prompt_text=it.prompt_text, gen_text=it.gen_text,
                method=args.method,
            )
            scores: Dict[str, float] = {}
            for k, m in metric_objs.items():
                try:
                    scores[k] = m.score(rec)
                except Exception as e:
                    log.warning("  metric=%s failed on %s: %s", k, rec.item_id, e)
                    scores[k] = float("nan")
            row = {
                "method": args.method, "item_id": rec.item_id,
                "pair_id": rec.pair_id, "pair_class": rec.pair_class,
                "L1": rec.L1, "L2": rec.L2,
                **scores,
            }
            fout.write(json.dumps(row) + "\n")
            per_item_rows.append(row)
            if (i + 1) % 20 == 0:
                log.info("[%d/%d] %s  %s  (elapsed=%.0fs)",
                         i + 1, len(available), rec.item_id,
                         {k: round(v, 3) for k, v in scores.items() if not math.isnan(v)},
                         time.time() - t0)

    # Aggregate
    summary = {
        "method": args.method,
        "n_items": len(per_item_rows),
        "n_available": len(available),
        "eval_set": str(args.eval_set),
        "gen_dir": str(gen_dir),
        "metric_notes": {
            "wer": "ASR edit error: WER for whitespace-tokenized languages; CER for zh/ja/ko.",
        },
        "generation": gen_summary,
        "by_pair": {},
        "overall": {},
        "by_class": {},
    }
    for key in [
        "ckpt", "K_test", "final_nfe", "no_meta_init", "use_adam",
        "no_phi", "no_cycle", "lid_only", "final_cfg_strength",
        "rerank_candidates", "rerank_scorer", "rerank_ecapa_weight",
        "rerank_asr_weight", "rerank_asr_topk", "rerank_asr_device",
        "rerank_asr_model_size", "rerank_asr_compute_type",
        "sampler",
    ]:
        if key in gen_summary:
            summary[key] = gen_summary[key]

    for metric in args.metrics:
        # overall
        summary["overall"][metric] = bootstrap_ci([r[metric] for r in per_item_rows])
        # by pair
        by_pair = defaultdict(list)
        by_class = defaultdict(list)
        for r in per_item_rows:
            by_pair[r["pair_id"]].append(r[metric])
            by_class[r["pair_class"]].append(r[metric])
        for pair, vals in by_pair.items():
            summary["by_pair"].setdefault(pair, {})[metric] = bootstrap_ci(vals)
        for cls, vals in by_class.items():
            summary["by_class"].setdefault(cls, {})[metric] = bootstrap_ci(vals)

    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("DONE %d items in %.0fs  ->  per-item=%s  summary=%s",
             len(per_item_rows), time.time() - t0, out_path, summary_path)


if __name__ == "__main__":
    main()
