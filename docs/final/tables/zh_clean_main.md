# Clean zh-expanded main table

| Method | Setting | SIM-o | ECAPA | ASR-Err | F0 | UTMOS |
| --- | --- | --- | --- | --- | --- | --- |
| F5-TTS | in-distrib | 0.901 | 0.574 | 0.966 | 0.004 | 1.539 |
| F5-TTS | zero-shot | 0.890 | 0.528 | 1.258 | 0.087 | 1.541 |
| F5 + verifier/ASR rerank | in-distrib | 0.921 | 0.596 | 0.757 | 0.012 | 1.586 |
| F5 + verifier/ASR rerank | zero-shot | 0.891 | 0.523 | 1.177 | 0.063 | 1.517 |
| CycleAdapt-Final | in-distrib | 0.919 | 0.599 | 0.888 | 0.018 | 1.594 |
| CycleAdapt-Final | zero-shot | 0.894 | 0.526 | 1.324 | 0.079 | 1.538 |
| Identity-only final | in-distrib | 0.921 | 0.595 | 0.947 | -0.010 | 1.566 |
| Identity-only final | zero-shot | 0.895 | 0.525 | 1.222 | 0.036 | 1.546 |
