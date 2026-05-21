# Runtime and Reproducibility

Runtime is an important limitation.

Clean expanded benchmark runtime:

| Method | Items | Seconds/item | Notes |
| --- | ---: | ---: | --- |
| F5-TTS | 175 | 1.34 | single generation |
| F5 + verifier/ASR rerank | 175 | 22.13 | 8 candidates + verifier rerank |
| CycleAdapt-Final | 175 | 21.99 | K-test 3 + 8 candidates + ASR top-k |
| Identity-only final | 175 | 20.95 | K-test 3 + 8 candidates + ASR top-k |

The current method is research-grade rather than deployment-grade.  The
important fairness point is that the strongest baselines use the same
multi-candidate reranking regime where applicable.

Tracked reproducibility artifacts:

- `checkpoints/*/final.pt`: final learned adapter/optimizer/loss-weighter
  checkpoints, small enough to keep in git.
- `results/eval_set.jsonl`: original clean eval split.
- `results/eval_set_zh_expanded.jsonl`: 175-item Chinese-source expanded split.
- `results/eval_set_zh_workshop_short3.jsonl`: 70-item short-prompt stress
  split.
- `results/eval_set_zh_workshop_noise10.jsonl`: 70-item noisy-prompt stress
  split.
- `results/scores/*.jsonl`: per-item metric outputs.
- `results/scores/*.summary.json`: aggregate score summaries.
- `results/tables/`: CSV/Markdown aggregates.

Still intentionally untracked:

- raw datasets and caches under `data/`;
- generated wav files under `results/audio/`;
- derived prompt wavs under `results/prompts/`;
- run logs under `logs/`;
- external F5/Vocos/WavLM/Whisper/SpeechBrain model caches.

Why this split matters: another agent can reproduce the exact evaluation splits
and use the trained CycleAdapt checkpoints, but the repository does not become
a data dump or a generated-audio archive.
