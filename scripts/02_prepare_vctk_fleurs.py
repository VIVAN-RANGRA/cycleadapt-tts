"""Stage 1b — extract VCTK + FLEURS tarballs into a flat manifest layout.

VCTK and FLEURS are downloaded as webdataset / tar.gz archives; we extract them
once to ``data/{vctk,fleurs}_extracted/`` so the manifest builder can index per
utterance.  AISHELL-3 and LibriTTS-R are read directly from the downloaded
parquet shards by ``cycle_tts.data``.

VCTK webdataset layout (per id):
    {id}.flac, {id}.normalized.txt, {id}.txt,
    {id}.speaker_id, {id}.accent, {id}.region, {id}.gender, {id}.age,
    {id}.txt_id

We extract everything and produce ``vctk_extracted/`` with the original files.
"""
from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Iterable

import tqdm

CYCLE_ROOT = Path(os.environ.get("CYCLE_TTS_ROOT", "/home/ubuntu/CYCLE_TTS"))
DATA = CYCLE_ROOT / "data"


def _safe_extractall(t: tarfile.TarFile, dst: Path) -> None:
    """Extract a tar without preserving the read-only / root-owned attrs that
    the upstream webdataset archives ship with (otherwise we cannot delete or
    overwrite the files afterwards as a non-root user).
    """
    dst.mkdir(parents=True, exist_ok=True)
    for member in t.getmembers():
        # Directories need execute (traverse); files need read-write.
        member.mode = 0o755 if member.isdir() else 0o644
        member.uid = os.getuid()
        member.gid = os.getgid()
        member.uname = ""
        member.gname = ""
    t.extractall(dst)


def extract_vctk() -> None:
    """VCTK webdataset shards all renumber from 0000000 (so tars collide).
    Extract each tar into its own subdirectory ``train-00/`` etc."""
    src_dir = DATA / "vctk_raw" / "audio"
    dst_root = DATA / "vctk_extracted"
    dst_root.mkdir(parents=True, exist_ok=True)
    tars = sorted(src_dir.glob("*.tar"))
    print(f"[vctk] extracting {len(tars)} tar files -> {dst_root}")
    for tar_path in tars:
        sub = dst_root / tar_path.stem
        if sub.exists() and any(sub.iterdir()):
            print(f"  {tar_path.name} (already extracted at {sub.name}, skipping)")
            continue
        sub.mkdir(parents=True, exist_ok=True)
        print(f"  {tar_path.name} -> {sub.name}/")
        with tarfile.open(tar_path) as t:
            _safe_extractall(t, sub)
    print(f"[vctk] done")


def extract_fleurs() -> None:
    """FLEURS arrives as nested .tar.gz with audio inside ``data/<lang>/audio/test.tar.gz``."""
    src_root = DATA / "fleurs" / "data"
    if not src_root.exists():
        print(f"[fleurs] no data dir at {src_root}, skipping")
        return
    for lang_dir in src_root.iterdir():
        if not lang_dir.is_dir():
            continue
        tgz = lang_dir / "audio" / "test.tar.gz"
        if not tgz.exists():
            continue
        dst = DATA / "fleurs_extracted" / lang_dir.name
        if dst.exists() and any(dst.iterdir()):
            print(f"[fleurs/{lang_dir.name}] already extracted, skipping")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        print(f"[fleurs/{lang_dir.name}] extracting -> {dst}")
        with tarfile.open(tgz) as t:
            _safe_extractall(t, dst)
    print("[fleurs] done")


def main() -> None:
    extract_vctk()
    extract_fleurs()
    print("[stage1b] all archives extracted")


if __name__ == "__main__":
    main()
