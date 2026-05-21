# Paired bootstrap significance

| Method | Baseline | Metric | n | mean_benefit | ci95_lo | ci95_hi | wins | win_rate | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CycleAdapt-Final | F5-TTS | simwavlm | 175 | 0.0083 | 0.0002 | 0.0176 | 95 | 0.5429 | positive means method is better; WER is sign-flipped |
| CycleAdapt-Final | F5-TTS | simecapa | 175 | 0.0056 | -0.0076 | 0.0183 | 91 | 0.5200 | positive means method is better; WER is sign-flipped |
| CycleAdapt-Final | F5-TTS | wer | 175 | -0.0253 | -0.1508 | 0.0959 | 78 | 0.4457 | positive means method is better; WER is sign-flipped |
| CycleAdapt-Final | F5-TTS | f0pcc | 144 | -0.0068 | -0.0647 | 0.0525 | 72 | 0.5000 | positive means method is better; WER is sign-flipped |
| CycleAdapt-Final | F5-TTS | utmos | 175 | 0.0135 | -0.0259 | 0.0529 | 91 | 0.5200 | positive means method is better; WER is sign-flipped |
| CycleAdapt-Final | F5 + verifier/ASR rerank | simwavlm | 175 | 0.0019 | -0.0004 | 0.0043 | 91 | 0.5200 | positive means method is better; WER is sign-flipped |
| CycleAdapt-Final | F5 + verifier/ASR rerank | simecapa | 175 | 0.0032 | -0.0023 | 0.0089 | 97 | 0.5543 | positive means method is better; WER is sign-flipped |
| CycleAdapt-Final | F5 + verifier/ASR rerank | wer | 175 | -0.1429 | -0.2546 | -0.0468 | 59 | 0.3371 | positive means method is better; WER is sign-flipped |
| CycleAdapt-Final | F5 + verifier/ASR rerank | f0pcc | 142 | 0.0045 | -0.0405 | 0.0491 | 75 | 0.5282 | positive means method is better; WER is sign-flipped |
| CycleAdapt-Final | F5 + verifier/ASR rerank | utmos | 174 | 0.0194 | -0.0024 | 0.0417 | 95 | 0.5460 | positive means method is better; WER is sign-flipped |
| Identity-only final | F5-TTS | simwavlm | 175 | 0.0093 | 0.0007 | 0.0191 | 100 | 0.5714 | positive means method is better; WER is sign-flipped |
| Identity-only final | F5-TTS | simecapa | 175 | 0.0044 | -0.0088 | 0.0175 | 92 | 0.5257 | positive means method is better; WER is sign-flipped |
| Identity-only final | F5-TTS | wer | 175 | 0.0312 | -0.0726 | 0.1355 | 85 | 0.4857 | positive means method is better; WER is sign-flipped |
| Identity-only final | F5-TTS | f0pcc | 143 | -0.0391 | -0.0940 | 0.0142 | 68 | 0.4755 | positive means method is better; WER is sign-flipped |
| Identity-only final | F5-TTS | utmos | 174 | 0.0097 | -0.0314 | 0.0530 | 92 | 0.5287 | positive means method is better; WER is sign-flipped |
| Identity-only final | F5 + verifier/ASR rerank | simwavlm | 175 | 0.0030 | 0.0003 | 0.0057 | 98 | 0.5600 | positive means method is better; WER is sign-flipped |
| Identity-only final | F5 + verifier/ASR rerank | simecapa | 175 | 0.0019 | -0.0048 | 0.0087 | 100 | 0.5714 | positive means method is better; WER is sign-flipped |
| Identity-only final | F5 + verifier/ASR rerank | wer | 175 | -0.0863 | -0.1705 | -0.0083 | 66 | 0.3771 | positive means method is better; WER is sign-flipped |
| Identity-only final | F5 + verifier/ASR rerank | f0pcc | 143 | -0.0186 | -0.0650 | 0.0278 | 70 | 0.4895 | positive means method is better; WER is sign-flipped |
| Identity-only final | F5 + verifier/ASR rerank | utmos | 174 | 0.0154 | -0.0089 | 0.0399 | 89 | 0.5115 | positive means method is better; WER is sign-flipped |
