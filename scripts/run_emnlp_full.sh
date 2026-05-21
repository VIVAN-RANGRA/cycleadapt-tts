#!/usr/bin/env bash
# EMNLP Findings full pipeline: fair NFE=32 eval + ablations + paper tables.
# Run after cycleadapt_emnlp_v3_fixed training completes (or pass a checkpoint).
set -u
set -o pipefail
cd /home/ubuntu/CYCLE_TTS
source scripts/env.sh

LOG=logs/runs
CKPT_V3="${1:-checkpoints/cycleadapt_emnlp_v3_fixed/final.pt}"
mkdir -p "$LOG" results/tables/emnlp

# Paper-fair synthesis settings (match B1 baseline).
NFE=32
CFG=2.0
# K_test=3 matches config default; --compile removed (slow with TTT, recompiles every step).
K_TEST=3
GEN_FLAGS=(--K-test "$K_TEST" --final-nfe "$NFE" --final-cfg-strength "$CFG")
OURS_FLAGS=("${GEN_FLAGS[@]}" --rerank-candidates 4)

run() {
  local n="$1"; shift
  echo "[$(date -u +%H:%M:%S)] START $n" | tee -a "$LOG/emnlp_full.log"
  if "$@" >"$LOG/${n}.log" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] DONE  $n" | tee -a "$LOG/emnlp_full.log"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL  $n" | tee -a "$LOG/emnlp_full.log"
    exit 1
  fi
}

[[ -f "$CKPT_V3" ]] || { echo "Missing $CKPT_V3 — wait for training."; exit 1; }

# B1 vanilla F5 at the exact paper-fair synthesis settings.
run b1_emnlp_gen python -u scripts/07_baseline_b1_f5.py \
  --out-dir results/audio/b1_f5_vanilla \
  --nfe "$NFE" --cfg-strength "$CFG" --overwrite
run b1_emnlp_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/b1_f5_vanilla --method b1_f5 \
  --out results/scores/b1_f5.jsonl

# B2 @ NFE=32 (no meta ckpt; random LoRA + Adam TTT).
run b2_emnlp_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "" --no-meta-init --use-adam --adam-lr 1e-3 \
  "${GEN_FLAGS[@]}" --overwrite --method-name b2_random_adam \
  --out-dir results/audio/b2_random_adam
run b2_emnlp_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/b2_random_adam --method b2_random_adam \
  --out results/scores/b2_random_adam.jsonl

# --- Adaptive methods @ NFE=32 (fair vs B1) ---
run ours_emnlp_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "$CKPT_V3" "${OURS_FLAGS[@]}" --overwrite --method-name ours_emnlp \
  --out-dir results/audio/ours_emnlp

run ours_emnlp_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/ours_emnlp --method ours_emnlp \
  --out results/scores/ours_emnlp.jsonl

run b3_emnlp_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "$CKPT_V3" "${GEN_FLAGS[@]}" --K-test 0 --overwrite --method-name b3_emnlp \
  --out-dir results/audio/b3_emnlp

run b3_emnlp_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/b3_emnlp --method b3_emnlp \
  --out results/scores/b3_emnlp.jsonl

run a1_no_phi_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "$CKPT_V3" "${OURS_FLAGS[@]}" --overwrite --no-phi --method-name a1_no_phi \
  --out-dir results/audio/a1_no_phi

run a1_no_phi_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/a1_no_phi --method a1_no_phi \
  --out results/scores/a1_no_phi.jsonl

run a3_no_cycle_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "$CKPT_V3" "${OURS_FLAGS[@]}" --overwrite --no-cycle --method-name a3_no_cycle \
  --out-dir results/audio/a3_no_cycle

run a3_no_cycle_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/a3_no_cycle --method a3_no_cycle \
  --out results/scores/a3_no_cycle.jsonl

run id_only_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "$CKPT_V3" "${OURS_FLAGS[@]}" --overwrite --id-only-ttt --method-name id_only_ttt \
  --out-dir results/audio/id_only_ttt

run id_only_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/id_only_ttt --method id_only_ttt \
  --out results/scores/id_only_ttt.jsonl

run emnlp_tables python -u scripts/12_aggregate_emnlp.py

python3 << 'PY'
import json
from pathlib import Path
print("\n======== EMNLP QUICK SUMMARY (zero-shot) ========")
for m in ["b1_f5", "b2_random_adam", "ours_emnlp", "b3_emnlp", "id_only_ttt", "a1_no_phi", "a3_no_cycle"]:
    p = Path(f"results/scores/{m}.summary.json")
    if not p.exists():
        print(f"  {m:18s}  pending")
        continue
    s = json.loads(p.read_text())
    zs = s["by_class"]["zero-shot"]["simwavlm"]["mean"]
    wer = s["by_class"]["zero-shot"]["wer"]["mean"]
    print(f"  {m:18s}  SIM-o={zs:.3f}  ASR-Err={wer:.3f}")
PY

echo "[$(date -u +%H:%M:%S)] EMNLP FULL PIPELINE DONE" | tee -a "$LOG/emnlp_full.log"
