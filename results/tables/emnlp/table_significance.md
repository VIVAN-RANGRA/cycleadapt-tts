# Wilcoxon vs cycleadapt_final (paired by item_id)

| baseline | metric | test |
|----------|--------|------|
| b1_f5 | simwavlm | p=0.000* (ours) |
| b1_f5 | wer | p=0.436 (ours) |
| b1_f5_rerank8 | simwavlm | p=0.013* (ours) |
| b1_f5_rerank8 | wer | p=0.982 (ours) |
| b2_random_adam | simwavlm | p=0.000* (ours) |
| b2_random_adam | wer | p=0.270 (ours) |
| b3_emnlp | simwavlm | p=0.000* (ours) |
| b3_emnlp | wer | p=0.941 (ours) |
| ours_emnlp | simwavlm | p=0.000* (ours) |
| ours_emnlp | wer | p=0.802 (ours) |
| cycleadapt_final_id | simwavlm | p=0.462 (other) |
| cycleadapt_final_id | wer | p=0.259 (ours) |
| a1_no_phi | simwavlm | p=0.001* (ours) |
| a1_no_phi | wer | p=0.872 (other) |
| a3_no_cycle | simwavlm | p=0.001* (ours) |
| a3_no_cycle | wer | p=0.864 (ours) |
| id_only_ttt | simwavlm | p=0.000* (ours) |
| id_only_ttt | wer | p=0.829 (other) |
