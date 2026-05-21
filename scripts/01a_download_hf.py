"""Stage 1a — download datasets via HuggingFace (fast, parallel).

Downloads:
  * VCTK            (badayvedat/VCTK)            -> data/vctk_raw/      ~8 GB    (110 EN speakers, eval)
  * AISHELL-3       (AISHELL/AISHELL-3)          -> data/aishell3_raw/  ~18 GB   (218 ZH speakers, meta-train)
  * LibriTTS-R      (mythicinfinity/libritts_r)  -> data/libritts_r/    ~30 GB   (EN, meta-train; train.clean.100 only)
  * Common Voice 17 ES + ZH                      -> data/common_voice/  via datasets.load_dataset, ~6 GB

Designed to run inside ``python -u``.  Each step is idempotent (HF caches files
by sha) and we drive multiple downloads concurrently with
``HfApi.snapshot_download(max_workers=16)``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

CYCLE_ROOT = Path(os.environ.get("CYCLE_TTS_ROOT", "/home/ubuntu/CYCLE_TTS"))
DATA = CYCLE_ROOT / "data"
HF_CACHE = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", DATA / "cache/huggingface/hub"))
DATA.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_vctk() -> None:
    out = DATA / "vctk_raw"
    out.mkdir(parents=True, exist_ok=True)
    log("VCTK -> downloading webdataset tars")
    t = time.time()
    snapshot_download(
        repo_id="badayvedat/VCTK",
        repo_type="dataset",
        local_dir=str(out),
        max_workers=8,
        allow_patterns=["audio/*.tar", "README.md"],
    )
    log(f"VCTK done in {time.time() - t:.0f}s -> {out}")


def download_aishell3() -> None:
    """MatrixStudio/AISHELL-3 packs the corpus as parquet shards (~400MB each)
    instead of the 88k individual wavs that the upstream AISHELL/AISHELL-3 repo
    serves.  At ~116 MB/s per shard from the HF CDN this finishes in ~4 min
    versus ~30 min via the per-file mirror.
    """
    out = DATA / "aishell3_raw"
    out.mkdir(parents=True, exist_ok=True)
    log("AISHELL-3 -> downloading parquet shards (MatrixStudio mirror)")
    t = time.time()
    snapshot_download(
        repo_id="MatrixStudio/AISHELL-3",
        repo_type="dataset",
        local_dir=str(out),
        max_workers=8,
        allow_patterns=["data/train-*.parquet", "data/test-*.parquet", "README.md"],
    )
    log(f"AISHELL-3 done in {time.time() - t:.0f}s -> {out}")


def download_libritts_r() -> None:
    out = DATA / "libritts_r"
    out.mkdir(parents=True, exist_ok=True)
    log("LibriTTS-R -> downloading train.clean.100 + dev.clean parquet shards")
    t = time.time()
    snapshot_download(
        repo_id="mythicinfinity/libritts_r",
        repo_type="dataset",
        local_dir=str(out),
        max_workers=8,
        allow_patterns=[
            "data/train.clean.100/*.parquet",
            "data/dev.clean/*.parquet",
            "README.md",
            "*.json",
        ],
    )
    log(f"LibriTTS-R done in {time.time() - t:.0f}s -> {out}")


def download_common_voice() -> None:
    """Common Voice ES + ZH for cross-lingual eval prompts.

    We use mozilla-foundation/common_voice_17_0; only download validated test
    splits (more than enough for our 30 eval speakers per language).
    """
    out = DATA / "common_voice"
    out.mkdir(parents=True, exist_ok=True)
    for lang in ["es", "zh-CN"]:
        log(f"Common Voice 17 / {lang} -> downloading test split")
        t = time.time()
        try:
            snapshot_download(
                repo_id="mozilla-foundation/common_voice_17_0",
                repo_type="dataset",
                local_dir=str(out),
                max_workers=8,
                allow_patterns=[
                    f"transcript/{lang}/test.tsv",
                    f"transcript/{lang}/validated.tsv",
                    f"audio/{lang}/test/*.tar",
                    f"audio/{lang}/validated/*.tar",
                ],
            )
            log(f"  {lang} done in {time.time() - t:.0f}s")
        except Exception as e:
            log(f"  CV {lang} failed: {e}")
            log("  Will fall back to FLEURS for that language at eval time.")


FLEURS_EVAL_LANGS = [
    "es_419",       # Spanish (LatAm)
    "cmn_hans_cn",  # Mandarin Chinese (Simplified)
    "fr_fr",        # French
    "de_de",        # German
    "hi_in",        # Hindi
    "ja_jp",        # Japanese
    "en_us",        # English (sanity / in-distribution)
]


def download_fleurs() -> None:
    """FLEURS for zero-shot eval across multiple languages.

    Each lang's test split is < 1 GB; downloading 6 is ~5 GB total.
    """
    out = DATA / "fleurs"
    out.mkdir(parents=True, exist_ok=True)
    log(f"FLEURS -> downloading test splits for {len(FLEURS_EVAL_LANGS)} langs: {FLEURS_EVAL_LANGS}")
    t = time.time()
    allow_patterns = []
    for lang in FLEURS_EVAL_LANGS:
        allow_patterns.extend([
            f"data/{lang}/audio/test.tar.gz",
            f"data/{lang}/test.tsv",
        ])
    try:
        snapshot_download(
            repo_id="google/fleurs",
            repo_type="dataset",
            local_dir=str(out),
            max_workers=8,
            allow_patterns=allow_patterns,
        )
        log(f"FLEURS done in {time.time() - t:.0f}s -> {out}")
    except Exception as e:
        log(f"FLEURS failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["vctk", "aishell3", "libritts_r", "fleurs", "common_voice"],
        choices=["vctk", "aishell3", "libritts_r", "common_voice", "fleurs"],
    )
    args = parser.parse_args()

    dispatch = {
        "vctk": download_vctk,
        "aishell3": download_aishell3,
        "libritts_r": download_libritts_r,
        "common_voice": download_common_voice,
        "fleurs": download_fleurs,
    }
    for name in args.datasets:
        try:
            dispatch[name]()
        except Exception as e:  # noqa: BLE001 — top-level guard
            log(f"[ERROR] {name} download failed: {type(e).__name__}: {e}")
            traceback_str = __import__("traceback").format_exc()
            log(traceback_str)
            sys.exit(1)
    log("All requested downloads finished.")


if __name__ == "__main__":
    main()
