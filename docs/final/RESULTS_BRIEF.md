# Results Brief

## Clean Chinese-Source Evaluation

The clean zh-expanded benchmark has 175 items across `zh->en`, `zh->zh`, and
zero-shot `zh->{es,fr,de,hi,ja}`.  All methods finished with `175/175` outputs.

Clean average results are modest: CycleAdapt improves speaker similarity over
vanilla F5-TTS, but does not decisively dominate the strong F5 + verifier/ASR
reranking baseline.  This should be framed honestly.

Key clean zero-shot numbers:

- F5-TTS: SIM-o 0.890, ASR-Err 1.258, UTMOS 1.541.
- F5 + verifier/ASR rerank: SIM-o 0.891, ASR-Err 1.177, UTMOS 1.517.
- CycleAdapt-Final: SIM-o 0.894, ASR-Err 1.324, UTMOS 1.538.
- Identity-only final: SIM-o 0.895, ASR-Err 1.222, UTMOS 1.546.

## Stress Tests

The strongest workshop evidence comes from prompt stress tests using a balanced
70-item subset per condition, 10 items per language pair.

Short 3-second prompts:

- F5-TTS zero-shot SIM-o: 0.722.
- F5 + rerank zero-shot SIM-o: 0.782.
- CycleAdapt-Final zero-shot SIM-o: 0.783.
- Identity-only final zero-shot SIM-o: 0.785.

Noisy 10 dB prompts:

- F5-TTS zero-shot SIM-o: 0.659, ASR-Err 1.131.
- F5 + rerank zero-shot SIM-o: 0.792, ASR-Err 1.274.
- CycleAdapt-Final zero-shot SIM-o: 0.799, ASR-Err 1.051.
- Identity-only final zero-shot SIM-o: 0.793, ASR-Err 1.182.

This gives a clean workshop claim: CycleAdapt is most useful when the prompt is
fragile.  In the noisy zero-shot condition, CycleAdapt-Final improves both
speaker preservation and ASR error over vanilla F5 and the reranked baseline.

## Caveat

The method is expensive.  Clean zh-expanded generation costs about 21-22 seconds
per item for CycleAdapt/reranked methods versus about 1.3 seconds for vanilla
F5-TTS.  The paper should state this plainly.
