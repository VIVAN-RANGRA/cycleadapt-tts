#!/usr/bin/env bash
# Master runner for the CycleAdapt-TTS EMNLP experiments.
#
# Strategy: use **internal baselines** that share the F5-TTS backbone (no
# third-party dependency conflicts) so the comparison is fair AND robust.
#
# Methods compared:
#   B1   Vanilla F5-TTS (no adapter, no TTT)
#   B2   F5 + random-init LoRA + standard Adam TTT (no meta-learning at all)
#   B3   F5 + meta-learned θ₀ but K_test=0 (just the learned init, no TTT)
#   ours Full CycleAdapt-TTS: meta-learned θ₀ + ψ + φ + K_test inner steps
#
# Ablations (after main training):
#   A1   ours with φ frozen at uniform
#   A3   ours with cycle losses disabled (only L_id)
#   A1-T training-time ablation: re-train without φ updates (~6h)
#
# Phases:
#   PHASE A  — runs in parallel with main training (uses spare GPU memory):
#              B1 (no TTT, fast), score B1, B2 (TTT with Adam, slower).
#   PHASE B  — runs once main checkpoint is written:
#              B3, ours, A1 (no-φ at test time), A3 (no-cycle at test time),
#              then score each.
#   PHASE C  — train no-φ ablation from scratch (~6h), generate & score.
#   PHASE D  — aggregate everything into the paper tables.

set -u
set -o pipefail

CYCLE_ROOT="/home/ubuntu/CYCLE_TTS"
cd "$CYCLE_ROOT"
source scripts/env.sh

LOG_DIR="$CYCLE_ROOT/logs/runs"
mkdir -p "$LOG_DIR"
RUN_NAME="cycleadapt_v1"
ABL_NAME="ablation_no_phi"
CKPT_DIR="$CYCLE_ROOT/checkpoints/$RUN_NAME"
ABL_CKPT_DIR="$CYCLE_ROOT/checkpoints/$ABL_NAME"

mark_started() { touch "$LOG_DIR/$1.started"; rm -f "$LOG_DIR/$1.done" "$LOG_DIR/$1.failed"; }
mark_done()    { touch "$LOG_DIR/$1.done"; }
mark_failed()  { touch "$LOG_DIR/$1.failed"; }

run_step() {
  local name="$1"; shift
  if [ -f "$LOG_DIR/$name.done" ]; then
    echo "[$(date -u +%H:%M:%S)] [run_all] SKIP  $name (already .done)" | tee -a "$LOG_DIR/master.log"
    return 0
  fi
  local logf="$LOG_DIR/$name.log"
  echo "[$(date -u +%H:%M:%S)] [run_all] START $name" | tee -a "$LOG_DIR/master.log"
  mark_started "$name"
  if ( "$@" ) >"$logf" 2>&1; then
    mark_done "$name"
    echo "[$(date -u +%H:%M:%S)] [run_all] DONE  $name" | tee -a "$LOG_DIR/master.log"
    return 0
  else
    mark_failed "$name"
    echo "[$(date -u +%H:%M:%S)] [run_all] FAIL  $name (see $logf)" | tee -a "$LOG_DIR/master.log"
    return 1
  fi
}

wait_for_file() {
  local fn="$1"
  echo "[$(date -u +%H:%M:%S)] [run_all] waiting for $fn ..." | tee -a "$LOG_DIR/master.log"
  while [ ! -f "$fn" ]; do sleep 60; done
  echo "[$(date -u +%H:%M:%S)] [run_all] $fn appeared." | tee -a "$LOG_DIR/master.log"
}

############################
# PHASE A — runs in parallel with the main training process.
# Only B1 (vanilla F5) goes here — it's forward-only and uses ~5 GB GPU.
# B2 needs the full feature-extractor stack (~10 GB additional) so it goes
# in Phase B after training releases the GPU.
############################

run_step "b1_f5_gen"      python -u scripts/07_baseline_b1_f5.py
run_step "b1_f5_score"    python -u scripts/09_score_method.py \
            --gen-dir results/audio/b1_f5_vanilla --method b1_f5 \
            --out results/scores/b1_f5.jsonl

############################
# PHASE B — wait for main training checkpoint, then run all
# adaptive baselines / ablations / our method.
############################
wait_for_file "$CKPT_DIR/final.pt"

# B2 — F5 + random LoRA + Adam TTT.  Doesn't need a checkpoint but reuses
# scripts/08 with --no-meta-init --use-adam.
run_step "b2_adam_gen"    python -u scripts/08_method_ours_ttt.py \
            --ckpt "" --no-meta-init --use-adam --adam-lr 1e-3 \
            --K-test 5 \
            --method-name b2_random_adam \
            --out-dir results/audio/b2_random_adam
run_step "b2_adam_score"  python -u scripts/09_score_method.py \
            --gen-dir results/audio/b2_random_adam --method b2_random_adam \
            --out results/scores/b2_random_adam.jsonl

# B3 — meta-learned init, K_test=0 (no inner adaptation).
run_step "b3_init_only_gen"   python -u scripts/08_method_ours_ttt.py \
            --ckpt "$CKPT_DIR/final.pt" --K-test 0 \
            --method-name b3_meta_init_only \
            --out-dir results/audio/b3_meta_init_only
run_step "b3_init_only_score" python -u scripts/09_score_method.py \
            --gen-dir results/audio/b3_meta_init_only --method b3_meta_init_only \
            --out results/scores/b3_meta_init_only.jsonl

# OURS — full method.
run_step "ours_gen"   python -u scripts/08_method_ours_ttt.py \
            --ckpt "$CKPT_DIR/final.pt" \
            --method-name ours \
            --out-dir results/audio/ours
run_step "ours_score" python -u scripts/09_score_method.py \
            --gen-dir results/audio/ours --method ours \
            --out results/scores/ours.jsonl

# Test-time ablation A1: same trained ckpt but freeze φ → uniform weights.
run_step "a1_no_phi_test_gen"   python -u scripts/08_method_ours_ttt.py \
            --ckpt "$CKPT_DIR/final.pt" --no-phi \
            --method-name a1_no_phi_test \
            --out-dir results/audio/a1_no_phi_test
run_step "a1_no_phi_test_score" python -u scripts/09_score_method.py \
            --gen-dir results/audio/a1_no_phi_test --method a1_no_phi_test \
            --out results/scores/a1_no_phi_test.jsonl

# Test-time ablation A3: same trained ckpt but disable cycle losses.
run_step "a3_no_cycle_gen"   python -u scripts/08_method_ours_ttt.py \
            --ckpt "$CKPT_DIR/final.pt" --no-cycle \
            --method-name a3_no_cycle \
            --out-dir results/audio/a3_no_cycle
run_step "a3_no_cycle_score" python -u scripts/09_score_method.py \
            --gen-dir results/audio/a3_no_cycle --method a3_no_cycle \
            --out results/scores/a3_no_cycle.jsonl

############################
# PHASE C — training-time ablation A1-T (no φ during meta-train).
############################
run_step "a1_train"  python -u scripts/10_ablation_no_phi.py \
            --M 1200 --B 4 --K 2 --run-name "$ABL_NAME"
wait_for_file "$ABL_CKPT_DIR/final.pt"
run_step "a1_train_gen"   python -u scripts/08_method_ours_ttt.py \
            --ckpt "$ABL_CKPT_DIR/final.pt" \
            --method-name a1_no_phi_trained \
            --out-dir results/audio/a1_no_phi_trained
run_step "a1_train_score" python -u scripts/09_score_method.py \
            --gen-dir results/audio/a1_no_phi_trained --method a1_no_phi_trained \
            --out results/scores/a1_no_phi_trained.jsonl

############################
# PHASE D — aggregate
############################
run_step "aggregate"  python -u scripts/11_aggregate_results.py

echo "[$(date -u +%H:%M:%S)] [run_all] ALL DONE" | tee -a "$LOG_DIR/master.log"
