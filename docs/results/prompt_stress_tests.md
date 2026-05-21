# Prompt Stress Test Results

The stress tests are the strongest current evidence for the method.  They ask
what happens when the prompt is fragile instead of clean:

- `short3`: reference prompt truncated to 3 seconds.
- `noise10`: reference prompt corrupted with 10 dB Gaussian noise.

Each stress condition uses 70 examples: 7 language pairs and 10 examples per
pair.

## Short 3s Prompts

Zero-shot summary:

| Method | SIM-o | ECAPA | ASR-Err | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| F5-TTS | 0.722 | 0.438 | 1.205 | 1.878 |
| F5 + verifier/ASR rerank | 0.782 | 0.445 | 1.054 | 1.763 |
| CycleAdapt-Final | 0.783 | 0.453 | 1.105 | 1.778 |
| Identity-only final | 0.785 | 0.444 | 1.080 | 1.775 |

Takeaway: adaptation closes most of the speaker-similarity gap caused by a
short prompt.  Identity-only has the best SIM-o, while full CycleAdapt has the
best ECAPA among zero-shot short-prompt methods.  Reranked F5 still has the best
ASR-Err in this condition.

## Noisy 10 dB Prompts

Zero-shot summary:

| Method | SIM-o | ECAPA | ASR-Err | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| F5-TTS | 0.659 | 0.198 | 1.131 | 1.141 |
| F5 + verifier/ASR rerank | 0.792 | 0.322 | 1.274 | 1.275 |
| CycleAdapt-Final | 0.799 | 0.326 | 1.051 | 1.249 |
| Identity-only final | 0.793 | 0.325 | 1.182 | 1.257 |

Takeaway: this is the best headline table.  CycleAdapt-Final has the best
zero-shot SIM-o and ECAPA, and it also has lower ASR-Err than both vanilla F5
and F5 + verifier/ASR rerank.  This is the condition where adaptation is not
just matching reranking; it is improving the candidate pool in a way the rerank
baseline alone does not capture.

Pair-level pattern:

- `zh_hi` under noise is a strong example for CycleAdapt-Final: SIM-o improves
  from 0.624 for vanilla F5 to 0.755.
- `zh_de`, `zh_fr`, and `zh_es` show large speaker-similarity recovery under
  noise.
- `zh_en` has a mixed ASR story for full CycleAdapt under noise; identity-only
  is better there.

The strongest paper claim should use the noisy-prompt result, then explain the
short-prompt result as corroborating evidence that adaptation is most useful
when the prompt is degraded.

Source tables:

- `results/tables/workshop_stress/stress_main.csv`
- `results/tables/workshop_stress/stress_by_pair.csv`
- `results/tables/workshop_stress/stress_deltas.csv`
- `docs/final/tables/stress_main.md`
- `docs/final/tables/stress_by_pair.md`
