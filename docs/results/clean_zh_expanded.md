# Clean Chinese-Source Expanded Results

The clean expanded benchmark uses 175 items: 25 examples for each of `zh_en`,
`zh_zh`, `zh_es`, `zh_fr`, `zh_de`, `zh_hi`, and `zh_ja`.  The source prompt is
Chinese and the target text spans in-distribution and zero-shot target
languages.

Main clean zero-shot results:

| Method | SIM-o | ECAPA | ASR-Err | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| F5-TTS | 0.890 | 0.528 | 1.258 | 1.541 |
| F5 + verifier/ASR rerank | 0.891 | 0.523 | 1.177 | 1.517 |
| CycleAdapt-Final | 0.894 | 0.526 | 1.324 | 1.538 |
| Identity-only final | 0.895 | 0.525 | 1.222 | 1.546 |

Interpretation:

- CycleAdapt and identity-only adaptation give small speaker-similarity gains
  over vanilla F5 on clean zero-shot prompts.
- F5 + verifier/ASR rerank is a hard baseline.  It improves ASR-Err while
  keeping speaker similarity competitive.
- Full CycleAdapt is not the clean-prompt winner because ASR-Err worsens
  relative to reranked F5.  The paper should say this plainly.
- Identity-only is the better clean zero-shot variant: it gets the highest
  SIM-o and UTMOS among the listed methods while avoiding the worst ASR penalty
  of the full cycle objective.

Language-pair notes:

- `zh_en` and `zh_zh` are the easiest settings.  Reranked F5 and adaptation
  both perform well.
- `zh_hi` and `zh_ja` are the important far-transfer settings.  CycleAdapt and
  identity-only help speaker similarity slightly, and identity-only is best for
  `zh_ja` ASR-Err.
- `zh_es` shows one of the cleaner speaker-similarity wins for CycleAdapt.
- `zh_de` and `zh_fr` expose the intelligibility tradeoff: SIM can improve
  while ASR-Err degrades.

The clean result should be framed as a diagnostic baseline, not the main
headline.  Its value is showing that prompt-only adaptation is competitive and
that the remaining issue is intelligibility alignment.

Source tables:

- `results/tables/zh_expanded/zh_table_main.csv`
- `results/tables/zh_expanded/zh_table_by_pair.csv`
- `results/tables/zh_expanded/zh_table_deltas.csv`
- `docs/final/tables/zh_clean_main.md`
- `docs/final/tables/zh_clean_by_pair.md`
