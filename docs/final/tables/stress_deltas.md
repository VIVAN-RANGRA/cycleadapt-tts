# Prompt stress paired deltas

| Condition | Method | Baseline | Metric | n | mean_benefit | wins |
| --- | --- | --- | --- | --- | --- | --- |
| short3 | CycleAdapt-Final | F5-TTS | simwavlm | 70 | +0.0582 | 50 |
| short3 | CycleAdapt-Final | F5-TTS | simecapa | 70 | +0.0078 | 37 |
| short3 | CycleAdapt-Final | F5-TTS | wer | 70 | +0.0862 | 27 |
| short3 | CycleAdapt-Final | F5-TTS | f0pcc | 44 | -0.1017 | 16 |
| short3 | CycleAdapt-Final | F5-TTS | utmos | 69 | -0.1205 | 28 |
| short3 | CycleAdapt-Final | F5 + verifier/ASR rerank | simwavlm | 70 | +0.0016 | 39 |
| short3 | CycleAdapt-Final | F5 + verifier/ASR rerank | simecapa | 70 | +0.0058 | 37 |
| short3 | CycleAdapt-Final | F5 + verifier/ASR rerank | wer | 70 | -0.0335 | 23 |
| short3 | CycleAdapt-Final | F5 + verifier/ASR rerank | f0pcc | 42 | +0.0555 | 25 |
| short3 | CycleAdapt-Final | F5 + verifier/ASR rerank | utmos | 69 | +0.0054 | 30 |
| short3 | Identity-only final | F5-TTS | simwavlm | 70 | +0.0585 | 52 |
| short3 | Identity-only final | F5-TTS | simecapa | 70 | +0.0014 | 35 |
| short3 | Identity-only final | F5-TTS | wer | 70 | +0.0863 | 26 |
| short3 | Identity-only final | F5-TTS | f0pcc | 43 | -0.0667 | 17 |
| short3 | Identity-only final | F5-TTS | utmos | 70 | -0.1287 | 25 |
| short3 | Identity-only final | F5 + verifier/ASR rerank | simwavlm | 70 | +0.0019 | 34 |
| short3 | Identity-only final | F5 + verifier/ASR rerank | simecapa | 70 | -0.0005 | 38 |
| short3 | Identity-only final | F5 + verifier/ASR rerank | wer | 70 | -0.0334 | 25 |
| short3 | Identity-only final | F5 + verifier/ASR rerank | f0pcc | 41 | +0.0883 | 23 |
| short3 | Identity-only final | F5 + verifier/ASR rerank | utmos | 69 | +0.0029 | 29 |
| noise10 | CycleAdapt-Final | F5-TTS | simwavlm | 70 | +0.1447 | 63 |
| noise10 | CycleAdapt-Final | F5-TTS | simecapa | 70 | +0.1369 | 63 |
| noise10 | CycleAdapt-Final | F5-TTS | wer | 70 | -0.0166 | 23 |
| noise10 | CycleAdapt-Final | F5-TTS | f0pcc | 52 | -0.0427 | 22 |
| noise10 | CycleAdapt-Final | F5-TTS | utmos | 63 | +0.1421 | 34 |
| noise10 | CycleAdapt-Final | F5 + verifier/ASR rerank | simwavlm | 70 | +0.0047 | 36 |
| noise10 | CycleAdapt-Final | F5 + verifier/ASR rerank | simecapa | 70 | +0.0016 | 34 |
| noise10 | CycleAdapt-Final | F5 + verifier/ASR rerank | wer | 70 | +0.0787 | 24 |
| noise10 | CycleAdapt-Final | F5 + verifier/ASR rerank | f0pcc | 60 | +0.0301 | 27 |
| noise10 | CycleAdapt-Final | F5 + verifier/ASR rerank | utmos | 66 | -0.0201 | 30 |
| noise10 | Identity-only final | F5-TTS | simwavlm | 70 | +0.1418 | 64 |
| noise10 | Identity-only final | F5-TTS | simecapa | 70 | +0.1350 | 62 |
| noise10 | Identity-only final | F5-TTS | wer | 70 | -0.0458 | 28 |
| noise10 | Identity-only final | F5-TTS | f0pcc | 53 | -0.0246 | 25 |
| noise10 | Identity-only final | F5-TTS | utmos | 63 | +0.1386 | 35 |
| noise10 | Identity-only final | F5 + verifier/ASR rerank | simwavlm | 70 | +0.0019 | 33 |
| noise10 | Identity-only final | F5 + verifier/ASR rerank | simecapa | 70 | -0.0003 | 44 |
| noise10 | Identity-only final | F5 + verifier/ASR rerank | wer | 70 | +0.0495 | 22 |
| noise10 | Identity-only final | F5 + verifier/ASR rerank | f0pcc | 61 | +0.0382 | 33 |
| noise10 | Identity-only final | F5 + verifier/ASR rerank | utmos | 67 | -0.0227 | 29 |
