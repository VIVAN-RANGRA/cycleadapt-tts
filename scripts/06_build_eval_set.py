#!/usr/bin/env python
"""Materialise the deterministic evaluation prompt+target set.

Run this once before launching baselines.  It writes::

    results/eval_set.jsonl

containing one ``EvalItem`` per line.  Baselines and our method both consume
this file so the comparison is fair.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path("/home/ubuntu/CYCLE_TTS")
sys.path.insert(0, str(ROOT / "src"))

from cycle_tts.config import CycleAdaptConfig  # noqa: E402
from cycle_tts.eval_prompts import build_eval_set, save_eval_set  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = CycleAdaptConfig()
    eval_pairs = list(cfg.eval_lang_pairs)
    logging.info("Building eval set for %d pairs:", len(eval_pairs))
    for L1, L2, cls in eval_pairs:
        logging.info("  %s -> %s  (%s)", L1, L2, cls)

    items = build_eval_set(eval_pairs, n_speakers=25, seed=cfg.train.seed)
    out_path = cfg.results_dir / "eval_set.jsonl"
    save_eval_set(items, out_path)
    logging.info("Wrote %d items -> %s", len(items), out_path)

    # Per-pair summary
    from collections import Counter
    by_pair = Counter((it.pair_id, it.pair_class) for it in items)
    for (pair_id, cls), n in sorted(by_pair.items()):
        logging.info("  %s [%s] : %d items", pair_id, cls, n)


if __name__ == "__main__":
    main()
