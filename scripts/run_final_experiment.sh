#!/usr/bin/env bash
# Final EMNLP-targeted experiment: verifier-guided identity adaptation.
set -u
set -o pipefail
cd /home/ubuntu/CYCLE_TTS
source scripts/env.sh

LOG=logs/runs
mkdir -p "$LOG" results/tables/emnlp

CKPT="${1:-checkpoints/cycleadapt_emnlp_v3_fixed/final.pt}"
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT"; exit 1; }

NFE=32
CFG=2.0
K_TEST=3
RERANK=8
RERANK_SCORER=wavlm_ecapa
ECAPA_W=0.3

GEN_FLAGS=(--final-nfe "$NFE" --final-cfg-strength "$CFG" \
  --rerank-candidates "$RERANK" --rerank-scorer "$RERANK_SCORER" \
  --rerank-ecapa-weight "$ECAPA_W")

run() {
  local n="$1"; shift
  echo "[$(date -u +%H:%M:%S)] START $n" | tee -a "$LOG/final_experiment.log"
  if "$@" >"$LOG/${n}.log" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] DONE  $n" | tee -a "$LOG/final_experiment.log"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL  $n" | tee -a "$LOG/final_experiment.log"
    tail -80 "$LOG/${n}.log" || true
    exit 1
  fi
}

# Strong no-adapter baseline with the same prompt-only verifier reranking.
run b1_rerank8_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "" --K-test 0 "${GEN_FLAGS[@]}" --overwrite \
  --method-name b1_f5_rerank8 --out-dir results/audio/b1_f5_rerank8
run b1_rerank8_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/b1_f5_rerank8 --method b1_f5_rerank8 \
  --out results/scores/b1_f5_rerank8.jsonl

# Main final method: fixed cycle/id objective, no learned phi collapse risk.
run final_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "$CKPT" --K-test "$K_TEST" "${GEN_FLAGS[@]}" --overwrite \
  --no-phi --method-name cycleadapt_final --out-dir results/audio/cycleadapt_final
run final_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/cycleadapt_final --method cycleadapt_final \
  --out results/scores/cycleadapt_final.jsonl

# Identity-only variant: keep it as a competitor/diagnostic in case it wins.
run final_id_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "$CKPT" --K-test "$K_TEST" "${GEN_FLAGS[@]}" --overwrite \
  --id-only-ttt --method-name cycleadapt_final_id --out-dir results/audio/cycleadapt_final_id
run final_id_score python -u scripts/09_score_method.py \
  --gen-dir results/audio/cycleadapt_final_id --method cycleadapt_final_id \
  --out results/scores/cycleadapt_final_id.jsonl

run final_tables python -u scripts/12_aggregate_emnlp.py

python3 << 'PY'
import json
from pathlib import Path
print("\n======== FINAL EXPERIMENT SUMMARY (zero-shot) ========")
for m in [
    "b1_f5", "b1_f5_rerank8", "b2_random_adam", "b3_emnlp",
    "ours_emnlp", "cycleadapt_final", "cycleadapt_final_id",
    "a1_no_phi", "a3_no_cycle", "id_only_ttt",
]:
    p = Path(f"results/scores/{m}.summary.json")
    if not p.exists():
        continue
    s = json.loads(p.read_text())
    z = s["by_class"]["zero-shot"]
    print(
        f"  {m:22s} SIM-o={z['simwavlm']['mean']:.3f} "
        f"ECAPA={z['simecapa']['mean']:.3f} "
        f"ASR-Err={z['wer']['mean']:.3f} "
        f"UTMOS={z['utmos']['mean']:.3f}"
    )
PY

echo "[$(date -u +%H:%M:%S)] FINAL EXPERIMENT DONE" | tee -a "$LOG/final_experiment.log"

