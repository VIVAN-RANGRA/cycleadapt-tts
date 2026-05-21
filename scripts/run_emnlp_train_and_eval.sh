#!/usr/bin/env bash
# Master launcher: v3 train (mel-cache + TF32; compile on eval only) then full EMNLP eval.
set -u
set -o pipefail
cd /home/ubuntu/CYCLE_TTS
source scripts/env.sh
LOG=logs/runs
mkdir -p "$LOG"

RUN=cycleadapt_emnlp_v3_fixed
CKPT="checkpoints/${RUN}/final.pt"
V2_INIT="checkpoints/cycleadapt_v2/final.pt"

echo "[$(date -u +%H:%M:%S)] === EMNLP v3 train (resume v2, mel-cache) ===" | tee "$LOG/emnlp_master.log"

# 600 extra iters on top of v2 weights ≈ full budget but faster per-step.
if [[ -f "$CKPT" ]]; then
  echo "Checkpoint exists: $CKPT — skipping train" | tee -a "$LOG/emnlp_master.log"
else
  if [[ -f "$V2_INIT" ]]; then
    RESUME=(--resume "$V2_INIT")
    echo "Warm-start from $V2_INIT" | tee -a "$LOG/emnlp_master.log"
  else
    RESUME=()
  fi
  # v2 final is step 799; train 600 *additional* meta-iters → M_total=1400.
  python -u scripts/05_meta_train.py \
    --run-name "$RUN" \
    --M 1400 --B 4 --K 2 \
    "${RESUME[@]}" \
    2>&1 | tee "$LOG/${RUN}_train.log"
fi

echo "[$(date -u +%H:%M:%S)] === EMNLP full eval ===" | tee -a "$LOG/emnlp_master.log"
bash scripts/run_emnlp_full.sh "$CKPT" 2>&1 | tee -a "$LOG/emnlp_master.log"
