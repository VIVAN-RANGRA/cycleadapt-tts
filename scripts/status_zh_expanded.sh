#!/usr/bin/env bash
set -u
cd /home/ubuntu/CYCLE_TTS

PID_FILE=logs/runs/zh_expanded_final_only_nohup.pid
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "zh-expanded final-only: RUNNING pid=$pid"
  else
    echo "zh-expanded final-only: NOT RUNNING last_pid=$pid"
  fi
else
  echo "zh-expanded final-only: no pid file"
fi

echo
echo "Stages:"
tail -n 20 logs/runs/zh_expanded_final_only.log 2>/dev/null || true

echo
echo "Fresh generation counts from timings.jsonl:"
for method in cycleadapt_zhx_final cycleadapt_zhx_final_id; do
  if [[ "$method" == "cycleadapt_zhx_final_id" ]] && ! grep -q "START final_zhx_id_gen" logs/runs/zh_expanded_final_only.log 2>/dev/null; then
    printf "%-28s %s\n" "$method" "pending (old timings ignored)"
    continue
  fi
  t="results/audio/${method}/timings.jsonl"
  if [[ -f "$t" ]]; then
    n=$(wc -l < "$t")
    if [[ "$n" == "0" ]]; then
      n=$(find "results/audio/${method}" -maxdepth 1 -type f -name '*.wav' -newer "$t" | wc -l)
      printf "%-28s %s/175 fresh wavs\n" "$method" "$n"
    else
      printf "%-28s %s/175\n" "$method" "$n"
    fi
  else
    printf "%-28s %s\n" "$method" "no timings yet"
  fi
done

echo
echo "Score summaries:"
python3 << 'PY'
import json
from pathlib import Path
for m in ["b1_f5_zhx", "b1_f5_zhx_rerank8_asr", "cycleadapt_zhx_final", "cycleadapt_zhx_final_id"]:
    p = Path(f"results/scores/{m}.summary.json")
    if not p.exists():
        print(f"{m:28s} missing")
        continue
    s = json.loads(p.read_text())
    n = s.get("n_available")
    z = s.get("by_class", {}).get("zero-shot")
    if z:
        print(f"{m:28s} n={n:>3} SIM={z['simwavlm']['mean']:.3f} ECAPA={z['simecapa']['mean']:.3f} ASR={z['wer']['mean']:.3f} UTMOS={z['utmos']['mean']:.3f}")
    else:
        print(f"{m:28s} n={n}")
PY

echo
echo "Recent generator log:"
tail -n 30 logs/runs/final_zhx_gen.log 2>/dev/null || true
