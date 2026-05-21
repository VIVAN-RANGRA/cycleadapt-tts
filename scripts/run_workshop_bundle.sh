#!/usr/bin/env bash
# Fast workshop evidence bundle: analysis + short/noisy prompt stress tests.
set -u
set -o pipefail
cd /home/ubuntu/CYCLE_TTS
source scripts/env.sh

LOG=logs/runs/workshop
mkdir -p "$LOG" results/tables/workshop results/tables/workshop_stress

CKPT="${1:-checkpoints/cycleadapt_emnlp_v3_fixed/final.pt}"
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT"; exit 1; }

N_PER_PAIR="${N_PER_PAIR:-10}"
NFE="${NFE:-32}"
CFG="${CFG:-2.0}"
K_TEST="${K_TEST:-3}"
RERANK="${RERANK:-8}"
ECAPA_W="${ECAPA_W:-0.3}"
ASR_W="${ASR_W:-0.05}"
ASR_TOPK="${ASR_TOPK:-2}"
ASR_DEVICE="${ASR_DEVICE:-cpu}"
ASR_MODEL="${ASR_MODEL:-tiny}"
ASR_COMPUTE="${ASR_COMPUTE:-int8}"

run() {
  local n="$1"; shift
  echo "[$(date -u +%H:%M:%S)] START $n" | tee -a "$LOG/workshop_bundle.log"
  if "$@" >"$LOG/${n}.log" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] DONE  $n" | tee -a "$LOG/workshop_bundle.log"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL  $n" | tee -a "$LOG/workshop_bundle.log"
    tail -80 "$LOG/${n}.log" || true
    exit 1
  fi
}

run workshop_analysis python -u scripts/14_workshop_analysis.py
run build_stress_eval python -u scripts/14_build_workshop_stress_eval.py --n-per-pair "$N_PER_PAIR"

for cond in short3 noise10; do
  eval_set="results/eval_set_zh_workshop_${cond}.jsonl"

  run "b1_${cond}_gen" python -u scripts/07_baseline_b1_f5.py \
    --eval-set "$eval_set" --out-dir "results/audio/b1_f5_zhx_${cond}" \
    --sampler eval --nfe "$NFE" --cfg-strength "$CFG" --overwrite
  run "b1_${cond}_score" python -u scripts/09_score_method.py \
    --eval-set "$eval_set" --gen-dir "results/audio/b1_f5_zhx_${cond}" \
    --method "b1_f5_zhx_${cond}" --out "results/scores/b1_f5_zhx_${cond}.jsonl"

  run "rerank_${cond}_gen" python -u scripts/08_method_ours_ttt.py \
    --ckpt "" --K-test 0 --eval-set "$eval_set" \
    --final-nfe "$NFE" --final-cfg-strength "$CFG" \
    --rerank-candidates "$RERANK" --rerank-scorer wavlm_ecapa_asr \
    --rerank-ecapa-weight "$ECAPA_W" --rerank-asr-weight "$ASR_W" \
    --rerank-asr-topk "$ASR_TOPK" --rerank-asr-device "$ASR_DEVICE" \
    --rerank-asr-model-size "$ASR_MODEL" --rerank-asr-compute-type "$ASR_COMPUTE" \
    --overwrite --method-name "b1_f5_zhx_rerank8_${cond}" \
    --out-dir "results/audio/b1_f5_zhx_rerank8_${cond}"
  run "rerank_${cond}_score" python -u scripts/09_score_method.py \
    --eval-set "$eval_set" --gen-dir "results/audio/b1_f5_zhx_rerank8_${cond}" \
    --method "b1_f5_zhx_rerank8_${cond}" --out "results/scores/b1_f5_zhx_rerank8_${cond}.jsonl"

  run "final_${cond}_gen" python -u scripts/08_method_ours_ttt.py \
    --ckpt "$CKPT" --K-test "$K_TEST" --eval-set "$eval_set" \
    --final-nfe "$NFE" --final-cfg-strength "$CFG" \
    --rerank-candidates "$RERANK" --rerank-scorer wavlm_ecapa_asr \
    --rerank-ecapa-weight "$ECAPA_W" --rerank-asr-weight "$ASR_W" \
    --rerank-asr-topk "$ASR_TOPK" --rerank-asr-device "$ASR_DEVICE" \
    --rerank-asr-model-size "$ASR_MODEL" --rerank-asr-compute-type "$ASR_COMPUTE" \
    --overwrite --no-phi --method-name "cycleadapt_zhx_final_${cond}" \
    --out-dir "results/audio/cycleadapt_zhx_final_${cond}"
  run "final_${cond}_score" python -u scripts/09_score_method.py \
    --eval-set "$eval_set" --gen-dir "results/audio/cycleadapt_zhx_final_${cond}" \
    --method "cycleadapt_zhx_final_${cond}" --out "results/scores/cycleadapt_zhx_final_${cond}.jsonl"

  run "id_${cond}_gen" python -u scripts/08_method_ours_ttt.py \
    --ckpt "$CKPT" --K-test "$K_TEST" --eval-set "$eval_set" \
    --final-nfe "$NFE" --final-cfg-strength "$CFG" \
    --rerank-candidates "$RERANK" --rerank-scorer wavlm_ecapa_asr \
    --rerank-ecapa-weight "$ECAPA_W" --rerank-asr-weight "$ASR_W" \
    --rerank-asr-topk "$ASR_TOPK" --rerank-asr-device "$ASR_DEVICE" \
    --rerank-asr-model-size "$ASR_MODEL" --rerank-asr-compute-type "$ASR_COMPUTE" \
    --overwrite --id-only-ttt --method-name "cycleadapt_zhx_final_id_${cond}" \
    --out-dir "results/audio/cycleadapt_zhx_final_id_${cond}"
  run "id_${cond}_score" python -u scripts/09_score_method.py \
    --eval-set "$eval_set" --gen-dir "results/audio/cycleadapt_zhx_final_id_${cond}" \
    --method "cycleadapt_zhx_final_id_${cond}" --out "results/scores/cycleadapt_zhx_final_id_${cond}.jsonl"
done

run aggregate_stress python -u scripts/16_aggregate_workshop_stress.py

echo "[$(date -u +%H:%M:%S)] WORKSHOP BUNDLE DONE" | tee -a "$LOG/workshop_bundle.log"
