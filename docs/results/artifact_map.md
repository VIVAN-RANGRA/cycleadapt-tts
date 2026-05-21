# Artifact Map

Use this file when deciding what to rerun or where a number came from.

## Checkpoints

Tracked final checkpoints:

- `checkpoints/cycleadapt_v1/final.pt`
- `checkpoints/cycleadapt_v2/final.pt`
- `checkpoints/cycleadapt_emnlp_v3/final.pt`
- `checkpoints/cycleadapt_emnlp_v3_fixed/final.pt`
- `checkpoints/proto_B4M5/final.pt`
- `checkpoints/proto_M3/final.pt`
- `checkpoints/proto_speed/final.pt`
- `checkpoints/proto_speed2/final.pt`

For paper results, the most important checkpoint is:

- `checkpoints/cycleadapt_emnlp_v3_fixed/final.pt`

`cycleadapt_v2/final.pt` is also important because the EMNLP v3 run warm-started
from it.

## Eval Splits

- `results/eval_set.jsonl`: original VCTK/FLEURS clean split.
- `results/eval_set_zh_expanded.jsonl`: 175-item Chinese-source multilingual
  split used for the main clean expanded benchmark.
- `results/eval_set_zh_workshop_short3.jsonl`: 70-item short-prompt stress
  split.
- `results/eval_set_zh_workshop_noise10.jsonl`: 70-item noisy-prompt stress
  split.

These JSONLs are tracked so future runs use exactly the same items.

## Generation Outputs

Generated wavs are not tracked.  Regenerate them with:

```bash
bash scripts/run_zh_expanded_experiment.sh
bash scripts/run_workshop_bundle.sh
```

Generated audio appears under `results/audio/`, which is intentionally ignored.

## Scores And Tables

Per-item scores:

- `results/scores/*.jsonl`

Aggregate score summaries:

- `results/scores/*.summary.json`

Paper tables:

- `results/tables/zh_expanded/`
- `results/tables/workshop/`
- `results/tables/workshop_stress/`
- `docs/final/tables/`

Paper figures:

- `docs/final/figures/`

Regenerate final materials with:

```bash
python scripts/17_make_final_materials.py
```
