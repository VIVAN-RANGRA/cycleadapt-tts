#!/usr/bin/env bash
# Stage 1a — direct downloads of evaluation/meta-training datasets that we keep
# on local disk.  Common Voice and MLS are pulled from the HuggingFace hub later
# via 01b_prepare_hf_subsets.py (no need to download the full archive).
#
# Total on-disk footprint after this script:
#   VCTK      ~11 GB   (eval, EN, 110 speakers)
#   AISHELL-3 ~18 GB   (meta-train, ZH, 218 speakers)
#   LibriTTS-R train-clean-100 ~7.5 GB (meta-train, EN, 247 speakers)
# Total ~37 GB.

set -euo pipefail

ROOT="${CYCLE_TTS_ROOT:-/home/ubuntu/CYCLE_TTS}"
DATA="${ROOT}/data"
mkdir -p "${DATA}/raw"

DL=("aria2c -x16 -s16 -j2 --auto-file-renaming=false --continue=true")
if ! command -v aria2c >/dev/null 2>&1; then
    DL=("wget -c --tries=5 --no-check-certificate")
fi

cd "${DATA}/raw"

# --- VCTK (eval) ---------------------------------------------------------
if [[ ! -d "${DATA}/vctk/wav48_silence_trimmed" ]]; then
    echo "[1a] downloading VCTK 0.92"
    if [[ ! -f "VCTK-Corpus-0.92.zip" ]]; then
        ${DL[@]} "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    fi
    unzip -q -o "VCTK-Corpus-0.92.zip" -d "${DATA}/vctk"
    echo "[1a] VCTK done"
else
    echo "[1a] VCTK already extracted, skipping"
fi

# --- AISHELL-3 (meta-train, ZH) -----------------------------------------
if [[ ! -d "${DATA}/aishell3/data_aishell3" ]]; then
    echo "[1a] downloading AISHELL-3"
    if [[ ! -f "data_aishell3.tgz" ]]; then
        ${DL[@]} "https://www.openslr.org/resources/93/data_aishell3.tgz"
    fi
    mkdir -p "${DATA}/aishell3"
    tar -xzf "data_aishell3.tgz" -C "${DATA}/aishell3"
    echo "[1a] AISHELL-3 done"
else
    echo "[1a] AISHELL-3 already extracted, skipping"
fi

# --- LibriTTS-R train-clean-100 (meta-train, EN) ------------------------
if [[ ! -d "${DATA}/libritts_r/LibriTTS_R/train-clean-100" ]]; then
    echo "[1a] downloading LibriTTS-R train-clean-100"
    if [[ ! -f "train_clean_100.tar.gz" ]]; then
        ${DL[@]} -o "train_clean_100.tar.gz" \
            "https://www.openslr.org/resources/141/train_clean_100.tar.gz"
    fi
    mkdir -p "${DATA}/libritts_r"
    tar -xzf "train_clean_100.tar.gz" -C "${DATA}/libritts_r"
    echo "[1a] LibriTTS-R done"
else
    echo "[1a] LibriTTS-R already extracted, skipping"
fi

echo "[1a] All direct downloads complete."
du -sh "${DATA}"/{vctk,aishell3,libritts_r} 2>/dev/null || true
