# Clean zh-expanded by language pair

| Pair | Method | n | SIM-o | ECAPA | ASR-Err | F0 | UTMOS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zh_de | F5-TTS | 25 | 0.894 | 0.550 | 1.366 | 0.022 | 1.618 |
| zh_de | F5 + verifier/ASR rerank | 25 | 0.892 | 0.562 | 1.229 | -0.017 | 1.596 |
| zh_de | CycleAdapt-Final | 25 | 0.896 | 0.559 | 1.627 | 0.051 | 1.606 |
| zh_de | Identity-only final | 25 | 0.896 | 0.556 | 1.407 | -0.048 | 1.612 |
| zh_en | F5-TTS | 25 | 0.888 | 0.541 | 0.951 | 0.070 | 1.593 |
| zh_en | F5 + verifier/ASR rerank | 25 | 0.915 | 0.570 | 0.761 | 0.065 | 1.583 |
| zh_en | CycleAdapt-Final | 25 | 0.914 | 0.576 | 0.938 | 0.036 | 1.570 |
| zh_en | Identity-only final | 25 | 0.916 | 0.567 | 1.158 | 0.023 | 1.582 |
| zh_es | F5-TTS | 25 | 0.895 | 0.553 | 1.134 | 0.138 | 1.566 |
| zh_es | F5 + verifier/ASR rerank | 25 | 0.909 | 0.554 | 1.049 | 0.084 | 1.524 |
| zh_es | CycleAdapt-Final | 25 | 0.912 | 0.561 | 1.078 | 0.121 | 1.515 |
| zh_es | Identity-only final | 25 | 0.911 | 0.563 | 1.133 | 0.067 | 1.556 |
| zh_fr | F5-TTS | 25 | 0.901 | 0.544 | 1.256 | 0.049 | 1.599 |
| zh_fr | F5 + verifier/ASR rerank | 25 | 0.902 | 0.538 | 1.106 | 0.206 | 1.538 |
| zh_fr | CycleAdapt-Final | 25 | 0.901 | 0.534 | 1.457 | 0.138 | 1.583 |
| zh_fr | Identity-only final | 25 | 0.901 | 0.544 | 1.188 | 0.103 | 1.575 |
| zh_hi | F5-TTS | 25 | 0.882 | 0.502 | 1.507 | 0.079 | 1.396 |
| zh_hi | F5 + verifier/ASR rerank | 25 | 0.878 | 0.480 | 1.447 | 0.012 | 1.354 |
| zh_hi | CycleAdapt-Final | 25 | 0.883 | 0.497 | 1.435 | 0.042 | 1.445 |
| zh_hi | Identity-only final | 25 | 0.884 | 0.492 | 1.433 | 0.020 | 1.441 |
| zh_ja | F5-TTS | 25 | 0.876 | 0.491 | 1.027 | 0.147 | 1.529 |
| zh_ja | F5 + verifier/ASR rerank | 25 | 0.874 | 0.479 | 1.054 | 0.022 | 1.575 |
| zh_ja | CycleAdapt-Final | 25 | 0.879 | 0.477 | 1.026 | 0.033 | 1.543 |
| zh_ja | Identity-only final | 25 | 0.882 | 0.472 | 0.947 | 0.029 | 1.551 |
| zh_zh | F5-TTS | 25 | 0.915 | 0.607 | 0.980 | -0.057 | 1.485 |
| zh_zh | F5 + verifier/ASR rerank | 25 | 0.926 | 0.621 | 0.753 | -0.044 | 1.590 |
| zh_zh | CycleAdapt-Final | 25 | 0.924 | 0.622 | 0.838 | -0.001 | 1.619 |
| zh_zh | Identity-only final | 25 | 0.926 | 0.624 | 0.737 | -0.047 | 1.551 |
