#!/usr/bin/env bash
# Evaluate cycleadapt_v2 on the full 250-item eval set.
set -u
set -o pipefail
cd /home/ubuntu/CYCLE_TTS
source scripts/env.sh

CKPT="checkpoints/cycleadapt_v2/final.pt"
LOG=logs/runs
mkdir -p "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date -u +%H:%M:%S)] START $name" | tee -a "$LOG/eval_v2.log"
  if "$@" >"$LOG/${name}.log" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] DONE  $name" | tee -a "$LOG/eval_v2.log"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL  $name" | tee -a "$LOG/eval_v2.log"
    exit 1
  fi
}

GEN=(--ckpt "$CKPT" --K-test 3 --final-nfe 16 --final-cfg-strength 2.0)

rm -rf results/audio/ours_v2 results/audio/b3_v2
rm -f results/scores/ours_v2.jsonl results/scores/ours_v2.summary.json \
      results/scores/b3_v2.jsonl results/scores/b3_v2.summary.json

run ours_v2_gen python -u scripts/08_method_ours_ttt.py \
  "${GEN[@]}" --method-name ours_v2 --out-dir results/audio/ours_v2

run ours_v2_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/ours_v2 --method ours_v2 --out results/scores/ours_v2.jsonl

run b3_v2_gen python -u scripts/08_method_ours_ttt.py \
  "${GEN[@]}" --K-test 0 --method-name b3_v2 --out-dir results/audio/b3_v2

run b3_v2_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/b3_v2 --method b3_v2 --out results/scores/b3_v2.jsonl

python3 -c "
import json
from pathlib import Path
print('=== v2 vs B1 (zero-shot SIM-o) ===')
for m in ['b1_f5', 'ours_v2', 'b3_v2']:
    p = Path(f'results/scores/{m}.summary.json')
    if not p.exists():
        print(m, 'missing'); continue
    s = json.loads(p.read_text())
    zs = s['by_class']['zero-shot']['simwavlm']['mean']
    id_ = s['by_class']['in-distrib']['simwavlm']['mean']
    print(f'{m:12s}  in-distrib={id_:.3f}  zero-shot={zs:.3f}')
"

run aggregate_v2 python -u scripts/11_aggregate_results.py

echo "[$(date -u +%H:%M:%S)] EVAL V2 DONE" | tee -a "$LOG/eval_v2.log"
