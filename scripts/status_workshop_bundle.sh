#!/usr/bin/env bash
set -u
cd /home/ubuntu/CYCLE_TTS

PID_FILE=logs/runs/workshop_bundle.pid
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "workshop bundle: RUNNING pid=$pid"
  else
    echo "workshop bundle: NOT RUNNING last_pid=$pid"
  fi
else
  echo "workshop bundle: no pid file"
fi

echo
echo "Stages:"
tail -n 30 logs/runs/workshop/workshop_bundle.log 2>/dev/null || true

echo
echo "Stress wav counts:"
for cond in short3 noise10; do
  for d in "b1_f5_zhx_${cond}" "b1_f5_zhx_rerank8_${cond}" "cycleadapt_zhx_final_${cond}" "cycleadapt_zhx_final_id_${cond}"; do
    printf "%-34s " "$d"
    find "results/audio/$d" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l
  done
done

echo
echo "Active processes:"
ps -ef | grep -E 'run_workshop_bundle|08_method_ours_ttt|09_score_method|07_baseline_b1_f5|16_aggregate_workshop' | grep -v grep || true
