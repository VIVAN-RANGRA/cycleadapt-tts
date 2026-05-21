#!/usr/bin/env bash
# Quick EMNLP pipeline status.
cd /home/ubuntu/CYCLE_TTS
echo "=== GPU / processes ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
ps aux | grep -E '05_meta_train|08_method|run_emnlp' | grep -v grep || echo "(no active jobs)"

echo ""
echo "=== Training v3 fixed ==="
CKPT=checkpoints/cycleadapt_emnlp_v3_fixed/final.pt
if [[ -f "$CKPT" ]]; then echo "DONE: $CKPT"; else
  tail -3 logs/runs/cycleadapt_emnlp_v3_fixed_train.log 2>/dev/null || echo "no train log yet"
fi

echo ""
echo "=== Scores (zero-shot SIM-o / ASR-Err) ==="
python3 << 'PY'
import json
from pathlib import Path
for m in ["b1_f5","b2_random_adam","ours_emnlp","b3_emnlp","id_only_ttt","a1_no_phi","a3_no_cycle"]:
    p = Path(f"results/scores/{m}.summary.json")
    if p.exists():
        s = json.loads(p.read_text())
        zs = s["by_class"]["zero-shot"]["simwavlm"]["mean"]
        wer = s["by_class"]["zero-shot"]["wer"]["mean"]
        print(f"  {m:18s} SIM-o={zs:.3f}  ASR-Err={wer:.3f}")
    else:
        print(f"  {m:18s} pending")
PY

echo ""
echo "=== Tables ==="
ls -la results/tables/emnlp/ 2>/dev/null || echo "  (not built yet)"
