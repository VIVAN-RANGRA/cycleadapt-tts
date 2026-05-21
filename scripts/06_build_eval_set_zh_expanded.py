#!/usr/bin/env python
"""Build an expanded Chinese-source evaluation set.

This keeps the original ``results/eval_set.jsonl`` untouched and writes:

    results/eval_set_zh_expanded.jsonl

Pairs cover Chinese prompts to all target languages available in the prepared
FLEURS manifest: English, Spanish, French, German, Hindi, Japanese, Chinese.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/ubuntu/CYCLE_TTS")
sys.path.insert(0, str(ROOT / "src"))

from cycle_tts.config import CycleAdaptConfig  # noqa: E402
from cycle_tts.eval_prompts import build_eval_set, save_eval_set  # noqa: E402


ZH_EXPANDED_PAIRS = [
    ("zh", "en", "in-distrib"),
    ("zh", "zh", "in-distrib"),
    ("zh", "es", "zero-shot"),
    ("zh", "fr", "zero-shot"),
    ("zh", "de", "zero-shot"),
    ("zh", "hi", "zero-shot"),
    ("zh", "ja", "zero-shot"),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = CycleAdaptConfig()
    logging.info("Building expanded zh-source eval set for %d pairs", len(ZH_EXPANDED_PAIRS))
    for L1, L2, cls in ZH_EXPANDED_PAIRS:
        logging.info("  %s -> %s  (%s)", L1, L2, cls)

    # Keep target utterances inside F5's 8192-frame DiT position limit. Hindi
    # uses more UTF-8 bytes per character, so unconstrained FLEURS samples can
    # create overlong generations for short Chinese prompts.
    items = build_eval_set(
        ZH_EXPANDED_PAIRS,
        n_speakers=25,
        seed=cfg.train.seed,
        target_max_utf8_bytes=300,
    )
    out_path = cfg.results_dir / "eval_set_zh_expanded.jsonl"
    save_eval_set(items, out_path)
    logging.info("Wrote %d items -> %s", len(items), out_path)

    by_pair = Counter((it.pair_id, it.pair_class) for it in items)
    for (pair_id, cls), n in sorted(by_pair.items()):
        logging.info("  %s [%s] : %d items", pair_id, cls, n)


if __name__ == "__main__":
    main()
