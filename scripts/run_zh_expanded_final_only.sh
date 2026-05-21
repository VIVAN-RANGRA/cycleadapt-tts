#!/usr/bin/env bash
# Re-run only the CycleAdapt Chinese-source final methods, then score/aggregate.
set -u
set -o pipefail
cd /home/ubuntu/CYCLE_TTS
source scripts/env.sh

LOG=logs/runs
mkdir -p "$LOG" results/tables/zh_expanded

CKPT="${1:-checkpoints/cycleadapt_emnlp_v3_fixed/final.pt}"
EVAL_SET=results/eval_set_zh_expanded.jsonl
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT"; exit 1; }
if [[ ! -f "$EVAL_SET" ]]; then
  python -u scripts/06_build_eval_set_zh_expanded.py || exit 1
fi

NFE=32
CFG=2.0
K_TEST=3
RERANK=8
ECAPA_W=0.3
ASR_W=0.05
ASR_TOPK="${ASR_TOPK:-2}"
ASR_DEVICE="${ASR_DEVICE:-cpu}"
ASR_MODEL="${ASR_MODEL:-tiny}"
ASR_COMPUTE="${ASR_COMPUTE:-int8}"

GEN_FLAGS=(--eval-set "$EVAL_SET" --final-nfe "$NFE" --final-cfg-strength "$CFG")
RERANK_FLAGS=("${GEN_FLAGS[@]}" --rerank-candidates "$RERANK" \
  --rerank-scorer wavlm_ecapa_asr --rerank-ecapa-weight "$ECAPA_W" \
  --rerank-asr-weight "$ASR_W" --rerank-asr-topk "$ASR_TOPK" \
  --rerank-asr-device "$ASR_DEVICE" \
  --rerank-asr-model-size "$ASR_MODEL" --rerank-asr-compute-type "$ASR_COMPUTE")

run() {
  local n="$1"; shift
  echo "[$(date -u +%H:%M:%S)] START $n" | tee -a "$LOG/zh_expanded_final_only.log"
  if "$@" >"$LOG/${n}.log" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] DONE  $n" | tee -a "$LOG/zh_expanded_final_only.log"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL  $n" | tee -a "$LOG/zh_expanded_final_only.log"
    tail -80 "$LOG/${n}.log" || true
    exit 1
  fi
}

run final_zhx_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "$CKPT" --K-test "$K_TEST" "${RERANK_FLAGS[@]}" --overwrite \
  --no-phi --method-name cycleadapt_zhx_final --out-dir results/audio/cycleadapt_zhx_final
run final_zhx_score python -u scripts/09_score_method.py \
  --eval-set "$EVAL_SET" --gen-dir results/audio/cycleadapt_zhx_final \
  --method cycleadapt_zhx_final --out results/scores/cycleadapt_zhx_final.jsonl

run final_zhx_id_gen python -u scripts/08_method_ours_ttt.py \
  --ckpt "$CKPT" --K-test "$K_TEST" "${RERANK_FLAGS[@]}" --overwrite \
  --id-only-ttt --method-name cycleadapt_zhx_final_id --out-dir results/audio/cycleadapt_zhx_final_id
run final_zhx_id_score python -u scripts/09_score_method.py \
  --eval-set "$EVAL_SET" --gen-dir results/audio/cycleadapt_zhx_final_id \
  --method cycleadapt_zhx_final_id --out results/scores/cycleadapt_zhx_final_id.jsonl

run zh_tables python -u scripts/13_aggregate_zh_expanded.py

python3 << 'PY'
import json
from pathlib import Path
for m in ["b1_f5_zhx", "b1_f5_zhx_rerank8_asr", "cycleadapt_zhx_final", "cycleadapt_zhx_final_id"]:
    p = Path(f"results/scores/{m}.summary.json")
    if not p.exists():
        continue
    s = json.loads(p.read_text())
    z = s["by_class"]["zero-shot"]
    print(f"{m:28s} ZH-zero SIM={z['simwavlm']['mean']:.3f} ECAPA={z['simecapa']['mean']:.3f} ASR={z['wer']['mean']:.3f} UTMOS={z['utmos']['mean']:.3f}")
PY

echo "[$(date -u +%H:%M:%S)] ZH FINAL-ONLY DONE" | tee -a "$LOG/zh_expanded_final_only.log"
