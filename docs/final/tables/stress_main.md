# Prompt stress main table

| Condition | Method | Setting | n | SIM-o | ECAPA | ASR-Err | F0 | UTMOS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short3 | F5-TTS | in-distrib | 20 | 0.705 | 0.443 | 0.479 | 0.255 | 2.098 |
| short3 | F5-TTS | zero-shot | 50 | 0.722 | 0.438 | 1.205 | 0.313 | 1.878 |
| short3 | F5 + verifier/ASR rerank | in-distrib | 20 | 0.754 | 0.433 | 0.438 | -0.034 | 1.943 |
| short3 | F5 + verifier/ASR rerank | zero-shot | 50 | 0.782 | 0.445 | 1.054 | 0.159 | 1.763 |
| short3 | CycleAdapt-Final | in-distrib | 20 | 0.756 | 0.435 | 0.427 | 0.056 | 1.924 |
| short3 | CycleAdapt-Final | zero-shot | 50 | 0.783 | 0.453 | 1.105 | 0.203 | 1.778 |
| short3 | Identity-only final | in-distrib | 20 | 0.753 | 0.434 | 0.490 | 0.052 | 1.905 |
| short3 | Identity-only final | zero-shot | 50 | 0.785 | 0.444 | 1.080 | 0.284 | 1.775 |
| noise10 | F5-TTS | in-distrib | 20 | 0.659 | 0.215 | 1.033 | -0.050 | 1.078 |
| noise10 | F5-TTS | zero-shot | 50 | 0.659 | 0.198 | 1.131 | 0.006 | 1.141 |
| noise10 | F5 + verifier/ASR rerank | in-distrib | 20 | 0.817 | 0.378 | 1.010 | 0.108 | 1.310 |
| noise10 | F5 + verifier/ASR rerank | zero-shot | 50 | 0.792 | 0.322 | 1.274 | -0.120 | 1.275 |
| noise10 | CycleAdapt-Final | in-distrib | 20 | 0.818 | 0.376 | 1.292 | 0.095 | 1.292 |
| noise10 | CycleAdapt-Final | zero-shot | 50 | 0.799 | 0.326 | 1.051 | -0.091 | 1.249 |
| noise10 | Identity-only final | in-distrib | 20 | 0.823 | 0.370 | 1.066 | 0.043 | 1.275 |
| noise10 | Identity-only final | zero-shot | 50 | 0.793 | 0.325 | 1.182 | -0.039 | 1.257 |
