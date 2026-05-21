"""Stage 1c — build train + eval manifests from extracted / parquet sources."""
from __future__ import annotations

import logging

from cycle_tts.data import build_all_manifests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main() -> None:
    counts = build_all_manifests()
    print("Manifest row counts:")
    for k, v in counts.items():
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
