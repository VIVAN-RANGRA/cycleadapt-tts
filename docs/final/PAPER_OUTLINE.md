# Workshop Paper Outline

## Title

Test-Time Speaker Adaptation for Cross-Lingual Voice Cloning: Benefits,
Tradeoffs, and Failure Modes

## Abstract

Emphasize prompt-only adaptation, cross-lingual F5-TTS, and robustness under
fragile prompt conditions.  Avoid claiming broad SOTA.

## 1. Introduction

- Cross-lingual voice cloning depends heavily on prompt quality.
- Modern F5-TTS is already strong under clean prompts.
- Test-time adaptation may help when prompt evidence is fragile.
- Main finding: average clean gains are modest, but stress gains are substantial.

## 2. Method

- Frozen F5-TTS backbone.
- Identity Alignment Adapter (LoRA).
- Learned/test-time updates on prompt-only losses.
- Variants: full CycleAdapt and identity-only adaptation.
- Verifier reranking applied consistently.

## 3. Experimental Setup

- Clean zh-expanded benchmark: 175 items, seven Chinese-source language pairs.
- Stress benchmark: balanced 70-item subsets for short and noisy prompt conditions.
- Metrics: WavLM SIM-o, ECAPA SIM, ASR-Err, UTMOS, F0 PCC.
- Baselines: F5-TTS and F5 + verifier/ASR reranking.

## 4. Results

- Clean setting: modest speaker gains, ASR tradeoff.
- Short prompt: Identity-only achieves best zero-shot SIM-o.
- Noisy prompt: CycleAdapt-Final achieves best zero-shot SIM-o and ASR-Err.
- Failure buckets: gains concentrate in low baseline-similarity cases.
- Runtime: adaptation is expensive.

## 5. Discussion

- Test-time adaptation helps when the prompt is unreliable.
- Reranking is a very strong baseline; adaptation should be framed as robustifying
  candidate generation, not replacing reranking.
- Intelligibility remains unresolved; future work should add differentiable or
  proxy ASR objectives inside adaptation.

## 6. Limitations

- Automatic metrics only, no listening panel.
- Stress tests use 70-item balanced subsets rather than full 175.
- CPU tiny ASR rerank is a pragmatic stability choice.
- Inference cost is high.
