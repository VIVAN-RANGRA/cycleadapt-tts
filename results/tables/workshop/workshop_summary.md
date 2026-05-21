# Workshop Analysis Summary

Generated from the completed zh-expanded experiment.

Files:
- `/home/ubuntu/CYCLE_TTS/results/tables/workshop/paired_significance.csv`: paired bootstrap confidence intervals and win rates.
- `/home/ubuntu/CYCLE_TTS/results/tables/workshop/failure_buckets.csv`: metrics by failure bucket and language pair.
- `/home/ubuntu/CYCLE_TTS/results/tables/workshop/bucket_deltas.csv`: bucket-level paired benefits.
- `/home/ubuntu/CYCLE_TTS/results/tables/workshop/runtime.csv`: runtime/cost table from generation summaries.

Primary reading guide:
- Treat positive `mean_benefit` as better; ASR error is sign-flipped.
- Use Identity-only final as the main stable CycleAdapt variant.
- Use failure buckets to support where adaptation helps, rather than overclaiming average SOTA.
