#!/usr/bin/env bash
# Source this to set up the CycleAdapt-TTS environment.
# Usage:  source scripts/env.sh

export CYCLE_TTS_ROOT="/home/ubuntu/CYCLE_TTS"
export HF_HOME="${CYCLE_TTS_ROOT}/data/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${CYCLE_TTS_ROOT}/data/cache/torch"
export MPLCONFIGDIR="${CYCLE_TTS_ROOT}/data/cache/matplotlib"
export NUMBA_CACHE_DIR="${CYCLE_TTS_ROOT}/data/cache/numba"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${CYCLE_TTS_ROOT}/src:${PYTHONPATH:-}"

# Aggressive CUDA allocator: use expandable segments and disable size-class
# rounding to avoid the fragmentation-related OOM we hit around iter ~660 in
# the first run.  See pytorch.org/docs/stable/notes/cuda.html.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

# Activate venv
source "${CYCLE_TTS_ROOT}/.venv/bin/activate"
