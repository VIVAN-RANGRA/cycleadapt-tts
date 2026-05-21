# Prompt stress by language pair

| Condition | Pair | Method | n | SIM-o | ECAPA | ASR-Err | F0 | UTMOS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short3 | zh_de | F5-TTS | 10 | 0.711 | 0.377 | 1.058 | 0.184 | 2.035 |
| short3 | zh_en | F5-TTS | 10 | 0.724 | 0.427 | 0.632 | 0.058 | 2.101 |
| short3 | zh_es | F5-TTS | 10 | 0.714 | 0.425 | 0.912 | 0.507 | 1.810 |
| short3 | zh_fr | F5-TTS | 10 | 0.751 | 0.445 | 1.025 | 0.365 | 1.968 |
| short3 | zh_hi | F5-TTS | 10 | 0.707 | 0.513 | 1.773 | 0.361 | 1.806 |
| short3 | zh_ja | F5-TTS | 10 | 0.727 | 0.432 | 1.256 | 0.149 | 1.769 |
| short3 | zh_zh | F5-TTS | 10 | 0.687 | 0.459 | 0.327 | 0.453 | 2.096 |
| short3 | zh_de | F5 + verifier/ASR rerank | 10 | 0.766 | 0.420 | 1.126 | 0.240 | 1.777 |
| short3 | zh_en | F5 + verifier/ASR rerank | 10 | 0.793 | 0.433 | 0.494 | -0.012 | 1.901 |
| short3 | zh_es | F5 + verifier/ASR rerank | 10 | 0.787 | 0.434 | 1.025 | 0.235 | 1.759 |
| short3 | zh_fr | F5 + verifier/ASR rerank | 10 | 0.803 | 0.433 | 0.974 | 0.159 | 1.884 |
| short3 | zh_hi | F5 + verifier/ASR rerank | 10 | 0.770 | 0.516 | 1.117 | -0.170 | 1.763 |
| short3 | zh_ja | F5 + verifier/ASR rerank | 10 | 0.783 | 0.424 | 1.030 | 0.270 | 1.632 |
| short3 | zh_zh | F5 + verifier/ASR rerank | 10 | 0.716 | 0.433 | 0.381 | -0.060 | 1.985 |
| short3 | zh_de | CycleAdapt-Final | 10 | 0.777 | 0.415 | 1.153 | 0.082 | 1.821 |
| short3 | zh_en | CycleAdapt-Final | 10 | 0.797 | 0.444 | 0.478 | 0.097 | 1.858 |
| short3 | zh_es | CycleAdapt-Final | 10 | 0.785 | 0.431 | 1.120 | 0.545 | 1.767 |
| short3 | zh_fr | CycleAdapt-Final | 10 | 0.808 | 0.437 | 0.986 | 0.116 | 1.861 |
| short3 | zh_hi | CycleAdapt-Final | 10 | 0.763 | 0.538 | 1.306 | 0.069 | 1.784 |
| short3 | zh_ja | CycleAdapt-Final | 10 | 0.784 | 0.441 | 0.962 | 0.158 | 1.660 |
| short3 | zh_zh | CycleAdapt-Final | 10 | 0.715 | 0.426 | 0.375 | -0.001 | 1.990 |
| short3 | zh_de | Identity-only final | 10 | 0.775 | 0.414 | 1.291 | 0.116 | 1.865 |
| short3 | zh_en | Identity-only final | 10 | 0.792 | 0.432 | 0.597 | 0.193 | 1.897 |
| short3 | zh_es | Identity-only final | 10 | 0.790 | 0.434 | 0.945 | 0.544 | 1.791 |
| short3 | zh_fr | Identity-only final | 10 | 0.805 | 0.443 | 0.989 | 0.180 | 1.850 |
| short3 | zh_hi | Identity-only final | 10 | 0.764 | 0.516 | 1.208 | 0.419 | 1.697 |
| short3 | zh_ja | Identity-only final | 10 | 0.789 | 0.413 | 0.968 | 0.137 | 1.672 |
| short3 | zh_zh | Identity-only final | 10 | 0.714 | 0.436 | 0.382 | -0.117 | 1.912 |
| noise10 | zh_de | F5-TTS | 10 | 0.676 | 0.189 | 1.358 | -0.019 | 1.194 |
| noise10 | zh_en | F5-TTS | 10 | 0.643 | 0.202 | 0.948 | -0.295 | 1.105 |
| noise10 | zh_es | F5-TTS | 10 | 0.661 | 0.197 | 1.018 | 0.072 | 1.107 |
| noise10 | zh_fr | F5-TTS | 10 | 0.665 | 0.236 | 0.996 | -0.117 | 1.222 |
| noise10 | zh_hi | F5-TTS | 10 | 0.624 | 0.150 | 1.341 | 0.116 | 1.087 |
| noise10 | zh_ja | F5-TTS | 10 | 0.672 | 0.219 | 0.944 | -0.029 | 1.102 |
| noise10 | zh_zh | F5-TTS | 10 | 0.676 | 0.227 | 1.119 | 0.257 | 1.054 |
| noise10 | zh_de | F5 + verifier/ASR rerank | 10 | 0.824 | 0.327 | 1.507 | 0.013 | 1.392 |
| noise10 | zh_en | F5 + verifier/ASR rerank | 10 | 0.789 | 0.374 | 1.050 | 0.207 | 1.272 |
| noise10 | zh_es | F5 + verifier/ASR rerank | 10 | 0.811 | 0.339 | 1.260 | -0.015 | 1.245 |
| noise10 | zh_fr | F5 + verifier/ASR rerank | 10 | 0.811 | 0.353 | 1.190 | 0.003 | 1.225 |
| noise10 | zh_hi | F5 + verifier/ASR rerank | 10 | 0.728 | 0.265 | 1.456 | -0.182 | 1.245 |
| noise10 | zh_ja | F5 + verifier/ASR rerank | 10 | 0.788 | 0.328 | 0.956 | -0.388 | 1.264 |
| noise10 | zh_zh | F5 + verifier/ASR rerank | 10 | 0.844 | 0.381 | 0.971 | 0.008 | 1.348 |
| noise10 | zh_de | CycleAdapt-Final | 10 | 0.820 | 0.339 | 1.073 | 0.063 | 1.351 |
| noise10 | zh_en | CycleAdapt-Final | 10 | 0.793 | 0.367 | 1.406 | 0.170 | 1.252 |
| noise10 | zh_es | CycleAdapt-Final | 10 | 0.810 | 0.327 | 0.959 | -0.078 | 1.259 |
| noise10 | zh_fr | CycleAdapt-Final | 10 | 0.815 | 0.342 | 1.086 | -0.035 | 1.210 |
| noise10 | zh_hi | CycleAdapt-Final | 10 | 0.755 | 0.276 | 1.160 | -0.078 | 1.164 |
| noise10 | zh_ja | CycleAdapt-Final | 10 | 0.793 | 0.343 | 0.977 | -0.286 | 1.268 |
| noise10 | zh_zh | CycleAdapt-Final | 10 | 0.842 | 0.384 | 1.178 | 0.021 | 1.332 |
| noise10 | zh_de | Identity-only final | 10 | 0.823 | 0.350 | 1.319 | 0.079 | 1.329 |
| noise10 | zh_en | Identity-only final | 10 | 0.802 | 0.355 | 1.015 | 0.104 | 1.246 |
| noise10 | zh_es | Identity-only final | 10 | 0.811 | 0.319 | 1.000 | 0.108 | 1.233 |
| noise10 | zh_fr | Identity-only final | 10 | 0.812 | 0.341 | 1.334 | -0.019 | 1.211 |
| noise10 | zh_hi | Identity-only final | 10 | 0.730 | 0.272 | 1.305 | -0.061 | 1.219 |
| noise10 | zh_ja | Identity-only final | 10 | 0.787 | 0.344 | 0.952 | -0.289 | 1.288 |
| noise10 | zh_zh | Identity-only final | 10 | 0.844 | 0.384 | 1.118 | -0.018 | 1.303 |
