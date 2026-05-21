#!/usr/bin/env bash
# Quick status dashboard for CycleAdapt-TTS runs.
LOG_DIR="/home/ubuntu/CYCLE_TTS/logs/runs"

echo "=========================================="
echo "  CycleAdapt-TTS run status  ($(date -u +%H:%M:%S))"
echo "=========================================="
echo
echo "## Main meta-training"
if [ -f "$LOG_DIR/cycleadapt_v1.log" ]; then
  tail -3 "$LOG_DIR/cycleadapt_v1.log" | sed 's/^/  /'
else
  echo "  (no log yet)"
fi
echo
echo "## Master runner"
if [ -f "$LOG_DIR/master.log" ]; then
  tail -10 "$LOG_DIR/master.log" | sed 's/^/  /'
else
  echo "  (master not started)"
fi
echo
echo "## Job markers"
ls -1 "$LOG_DIR"/*.started 2>/dev/null | sed 's|.*/||; s/\.started$//' | sort > /tmp/started.list
ls -1 "$LOG_DIR"/*.done    2>/dev/null | sed 's|.*/||; s/\.done$//'    | sort > /tmp/done.list
ls -1 "$LOG_DIR"/*.failed  2>/dev/null | sed 's|.*/||; s/\.failed$//'  | sort > /tmp/failed.list
echo "  started: $(wc -l < /tmp/started.list)"
echo "  done:    $(wc -l < /tmp/done.list)"
echo "  failed:  $(wc -l < /tmp/failed.list)"
echo
echo "## Detailed:"
for s in $(cat /tmp/started.list); do
  if grep -qFx "$s" /tmp/done.list; then     state="DONE   "
  elif grep -qFx "$s" /tmp/failed.list; then state="FAILED "
  else                                       state="RUNNING"
  fi
  echo "  $state  $s"
done
echo
echo "## GPU"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'
echo
echo "## Disk"
du -sh /home/ubuntu/CYCLE_TTS/results/audio/*/  2>/dev/null | sed 's|^|  |'
