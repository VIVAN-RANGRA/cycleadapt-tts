# Results Dossier

This directory is the readable experiment dossier.  `docs/final/` is optimized
for paper writing, while this folder explains what each result means and how to
argue it.

Recommended reading order:

1. `clean_zh_expanded.md`
2. `prompt_stress_tests.md`
3. `failure_buckets_and_ablations.md`
4. `runtime_and_reproducibility.md`
5. `paper_claims_limitations.md`
6. `artifact_map.md`

Short version: CycleAdapt is not a universal clean-prompt replacement for a
strong F5 + reranking baseline.  Its best evidence is under fragile prompts:
short or noisy references, where prompt-only adaptation recovers speaker
similarity and, in the noisy setting, improves ASR error over reranked F5.
