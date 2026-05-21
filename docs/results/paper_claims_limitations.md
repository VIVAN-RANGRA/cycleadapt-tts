# Paper Claims and Limitations

## Best Claim

CycleAdapt is a prompt-only test-time adaptation method for cross-lingual voice
cloning.  It is most useful when the source prompt is fragile.  On noisy
Chinese-source zero-shot transfer, CycleAdapt-Final improves speaker similarity
over both vanilla F5 and a strong F5 reranking baseline while also reducing
ASR-Err relative to that reranked baseline.

## Safe Workshop Framing

This is a strong workshop paper if framed as:

> Test-time speaker adaptation for cross-lingual voice cloning under fragile
> prompts: benefits, tradeoffs, and failure modes.

It is not yet a strong Findings/SOTA paper because:

- Clean-prompt gains are small.
- The F5 + verifier/ASR rerank baseline is extremely competitive.
- Intelligibility is not consistently improved.
- Runtime is high because adaptation and reranking are both expensive.
- The method has not yet been validated with a listening panel.

## What Not To Claim

Do not claim:

- universal SOTA over all baselines;
- consistent ASR improvements across all languages;
- that full CycleAdapt is always better than identity-only;
- that average clean-prompt results alone justify the method.

## What To Emphasize

Emphasize:

- stress robustness;
- low-baseline-similarity buckets;
- fair comparison against reranked F5;
- transparent negative results;
- the method as an adaptation layer that can improve candidate quality before
  or alongside reranking.

## Next Experiments For A Stronger Paper

Highest-value additions:

1. Add an intelligibility proxy to adaptation: multilingual ASR, CTC/text
   consistency, Whisper encoder agreement, or phoneme consistency.
2. Meta-train on the actual target regime: multilingual support episodes for
   Hindi, Japanese, French, German, and Spanish, not only English/Chinese.
3. Run larger stress subsets beyond 70 items per condition.
4. Add prompt transcript mismatch and no-prompt-transcript settings.
5. Add a small listening panel once the objective story is stable.
