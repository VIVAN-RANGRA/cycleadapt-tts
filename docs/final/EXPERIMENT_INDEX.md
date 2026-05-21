# Experiment Index

## Main Clean zh-expanded Experiment

Purpose: Chinese-source cross-lingual evaluation on clean prompts.

Eval set:
- Built by `scripts/06_build_eval_set_zh_expanded.py`.
- 175 total items, 25 per language pair.
- Pairs: `zh_en`, `zh_zh`, `zh_es`, `zh_fr`, `zh_de`, `zh_hi`, `zh_ja`.

Methods:
- `b1_f5_zhx`: vanilla F5-TTS.
- `b1_f5_zhx_rerank8_asr`: F5 with verifier/ASR reranking.
- `cycleadapt_zhx_final`: CycleAdapt-Final.
- `cycleadapt_zhx_final_id`: identity-only adaptation.

Key scripts:
- `scripts/run_zh_expanded_experiment.sh`: full baseline + final run.
- `scripts/run_zh_expanded_final_only.sh`: re-run only CycleAdapt final methods.
- `scripts/13_aggregate_zh_expanded.py`: aggregate tables.

Important tables:
- `results/tables/zh_expanded/zh_table_main.csv`
- `results/tables/zh_expanded/zh_table_by_pair.csv`
- `results/tables/zh_expanded/zh_table_deltas.csv`

## Workshop Analysis

Purpose: paired significance, failure buckets, runtime/cost.

Scripts:
- `scripts/14_workshop_analysis.py`

Tables:
- `results/tables/workshop/paired_significance.csv`
- `results/tables/workshop/failure_buckets.csv`
- `results/tables/workshop/bucket_deltas.csv`
- `results/tables/workshop/runtime.csv`

## Prompt Stress Tests

Purpose: evaluate short/noisy prompt robustness.

Eval sets:
- Built by `scripts/14_build_workshop_stress_eval.py`.
- `short3`: prompt truncated to 3 seconds.
- `noise10`: prompt corrupted with 10 dB Gaussian noise.
- 70 items per condition, balanced across seven language pairs.

Runner:
- `scripts/run_workshop_bundle.sh`

Aggregator:
- `scripts/16_aggregate_workshop_stress.py`

Tables:
- `results/tables/workshop_stress/stress_main.csv`
- `results/tables/workshop_stress/stress_by_pair.csv`
- `results/tables/workshop_stress/stress_deltas.csv`
