#!/usr/bin/env bash
# Fast re-eval from last good checkpoint (before NaN at iter ~1305).
# Uses K_test=2, final_nfe=16 (~2x faster than default K=3, NFE=32).
set -u
set -o pipefail
cd /home/ubuntu/CYCLE_TTS
source scripts/env.sh

LOG=logs/runs
CKPT="checkpoints/cycleadapt_v1/step001200.pt"
mkdir -p "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date -u +%H:%M:%S)] START $name" | tee -a "$LOG/fast_reeval.log"
  if "$@" >"$LOG/${name}.log" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] DONE  $name" | tee -a "$LOG/fast_reeval.log"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL  $name" | tee -a "$LOG/fast_reeval.log"
    return 1
  fi
}

# Symlink good checkpoint for clarity
ln -sf "$(basename "$CKPT")" checkpoints/cycleadapt_v1/best_before_nan.pt

FAST_GEN=(--ckpt "$CKPT" --K-test 2 --final-nfe 16 --final-cfg-strength 2.0)

# Clear stale broken outputs so we don't skip by mistake
for d in ours b3_meta_init_only a1_no_phi_test a3_no_cycle; do
  rm -rf "results/audio/${d}"
done
rm -f results/scores/ours.jsonl results/scores/ours.summary.json \
      results/scores/b3_meta_init_only.jsonl results/scores/b3_meta_init_only.summary.json \
      results/scores/a1_no_phi_test.jsonl results/scores/a1_no_phi_test.summary.json \
      results/scores/a3_no_cycle.jsonl results/scores/a3_no_cycle.summary.json

run ours_gen python -u scripts/08_method_ours_ttt.py \
  "${FAST_GEN[@]}" --method-name ours --out-dir results/audio/ours

run ours_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/ours --method ours --out results/scores/ours.jsonl

run b3_gen python -u scripts/08_method_ours_ttt.py \
  "${FAST_GEN[@]}" --K-test 0 --method-name b3_meta_init_only \
  --out-dir results/audio/b3_meta_init_only

run b3_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/b3_meta_init_only --method b3_meta_init_only \
  --out results/scores/b3_meta_init_only.jsonl

run a1_gen python -u scripts/08_method_ours_ttt.py \
  "${FAST_GEN[@]}" --no-phi --method-name a1_no_phi_test \
  --out-dir results/audio/a1_no_phi_test

run a1_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/a1_no_phi_test --method a1_no_phi_test \
  --out results/scores/a1_no_phi_test.jsonl

run a3_gen python -u scripts/08_method_ours_ttt.py \
  "${FAST_GEN[@]}" --no-cycle --method-name a3_no_cycle \
  --out-dir results/audio/a3_no_cycle

run a3_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/a3_no_cycle --method a3_no_cycle \
  --out results/scores/a3_no_cycle.jsonl

# Quick sanity: print ours vs b1 SIM-o
python3 -c "
import json
from pathlib import Path
for m in ['b1_f5','ours','b3_meta_init_only']:
    p=Path(f'results/scores/{m}.summary.json')
    if not p.exists(): continue
    s=json.loads(p.read_text())
    zs=s.get('by_class',{}).get('zero-shot',{}).get('simwavlm',{}).get('mean',float('nan'))
    print(f'{m:20s} zero-shot SIM-o={zs:.3f}')
"

run aggregate python -u scripts/11_aggregate_results.py

echo "[$(date -u +%H:%M:%S)] FAST REEVAL ALL DONE" | tee -a "$LOG/fast_reeval.log"
