# Failure Buckets and Ablations

The average clean result hides the useful behavior.  The bucket analysis shows
that CycleAdapt is most valuable when vanilla F5 begins from a weaker speaker
match.

Important buckets:

- `low_f5_wavlm_q25`: bottom quartile under F5 WavLM speaker similarity.
- `low_f5_ecapa_q25`: bottom quartile under F5 ECAPA similarity.
- `high_f5_asr_q75`: high ASR-error examples under F5.
- `far_zh_hi_ja`: far-transfer languages Hindi and Japanese.

Main bucket observations:

- In `low_f5_wavlm_q25`, CycleAdapt-Final improves SIM-o over F5 by about
  +0.048, and identity-only improves by about +0.050.
- In `low_f5_ecapa_q25`, identity-only has the strongest speaker result among
  the tested variants.
- In `far_zh_hi_ja`, identity-only is the cleanest variant: it gives better
  SIM-o than F5 and lower ASR-Err than full CycleAdapt.
- In `high_f5_asr_q75`, reranking helps ASR substantially; adaptation alone
  does not fully solve intelligibility.

Variant interpretation:

- Full CycleAdapt is useful under noise because the cycle losses appear to help
  stabilize the prompt-conditioned update when the prompt is corrupted.
- Identity-only is often better on clean and far-transfer settings because it
  avoids noisy cross-lingual reconstruction/cycle signals.
- `phi` learned loss weighting needed an active-loss mask.  Without the mask,
  it could put probability mass on inactive zero losses and create misleadingly
  low objectives.
- Random Adam / untrained adapter baselines are important negative controls:
  improvements should be attributed to learned initialization and stable
  adaptation, not merely extra generation attempts.

What to write:

> CycleAdapt is not uniformly better on every average metric.  Its strength is
> conditional: it helps when the prompt is short, noisy, or when the initial F5
> speaker match is weak.  The ablations suggest that identity preservation is
> the reliable part of the objective, while intelligibility-aware adaptation is
> still missing.

Source tables:

- `results/tables/workshop/failure_buckets.csv`
- `results/tables/workshop/bucket_deltas.csv`
- `results/tables/workshop/paired_significance.csv`
- `docs/final/tables/failure_buckets.md`
- `docs/final/tables/bucket_deltas.md`
