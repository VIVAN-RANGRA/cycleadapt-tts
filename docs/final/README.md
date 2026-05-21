# Final Paper Materials

This folder is the compact source of truth for writing the workshop paper.
It contains the tables, plots, and interpretation distilled from the completed
CycleAdapt-TTS experiments.  Large artifacts such as datasets, generated audio,
checkpoints, and logs are intentionally not included in git.

## Strongest Takeaway

CycleAdapt is not a clean SOTA replacement for verifier reranking on clean
prompts.  Its stronger contribution is robustness: under short or noisy prompt
conditions, adaptation substantially improves speaker preservation over vanilla
F5-TTS, and under noisy zero-shot transfer CycleAdapt-Final improves both
speaker similarity and ASR error relative to vanilla F5 and the reranked F5
baseline.

## Recommended Paper Claim

> Prompt-only test-time adaptation is useful for fragile cross-lingual voice
> cloning conditions.  Identity-only and CycleAdapt variants improve speaker
> preservation under short/noisy prompts, while intelligibility remains the key
> tradeoff and requires explicit ASR-aware treatment.

## Contents

- `RESULTS_BRIEF.md`: short narrative for the results section.
- `EXPERIMENT_INDEX.md`: what was run and where outputs live.
- `PAPER_OUTLINE.md`: workshop paper skeleton.
- `tables/`: Markdown and CSV copies of paper-ready tables.
- `figures/`: PNG plots for the paper draft.
