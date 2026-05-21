# Significance tests (Wilcoxon signed-rank, paired by item_id, vs. `ours`)

| method | metric | n | W | p | direction |
|--------|--------|---|--:|---:|-----------|
| a1_no_phi_test | simwavlm | 250 | 13504.0 | 5.64e-02 | ours better |
| a1_no_phi_test | simecapa | 250 | 15023.0 | 5.62e-01 | a1_no_phi_test better |
| a1_no_phi_test | f0pcc | 248 | 14881.0 | 6.22e-01 | a1_no_phi_test better |
| a1_no_phi_test | wer | 250 | 15209.0 | 6.40e-01 | a1_no_phi_test better |
| a1_no_phi_test | utmos | 250 | 14640.0 | 3.60e-01 | a1_no_phi_test better |
| a3_no_cycle | simwavlm | 250 | 14758.0 | 4.17e-01 | ours better |
| a3_no_cycle | simecapa | 250 | 15122.0 | 6.21e-01 | ours better |
| a3_no_cycle | f0pcc | 248 | 15328.0 | 9.23e-01 | ours better |
| a3_no_cycle | wer | 250 | 15342.0 | 7.36e-01 | a3_no_cycle better |
| a3_no_cycle | utmos | 250 | 13293.0 | 3.64e-02 | a3_no_cycle better |
| b1_f5 | simwavlm | 250 | 6.0 | 9.95e-43 | b1_f5 better |
| b1_f5 | simecapa | 250 | 0.0 | 9.25e-43 | b1_f5 better |
| b1_f5 | f0pcc | 229 | 10549.0 | 9.08e-03 | b1_f5 better |
| b1_f5 | wer | 250 | 9028.5 | 5.91e-09 | b1_f5 better |
| b1_f5 | utmos | 0 | — | — | — |
| b2_random_adam | simwavlm | 250 | 5.0 | 9.83e-43 | b2_random_adam better |
| b2_random_adam | simecapa | 250 | 0.0 | 9.25e-43 | b2_random_adam better |
| b2_random_adam | f0pcc | 220 | 9577.0 | 6.38e-03 | b2_random_adam better |
| b2_random_adam | wer | 250 | 8600.5 | 5.90e-10 | b2_random_adam better |
| b2_random_adam | utmos | 248 | 123.0 | 8.70e-42 | b2_random_adam better |
| b3_meta_init_only | simwavlm | 250 | 12281.0 | 2.92e-03 | ours better |
| b3_meta_init_only | simecapa | 250 | 15322.0 | 7.49e-01 | b3_meta_init_only better |
| b3_meta_init_only | f0pcc | 247 | 14054.0 | 2.62e-01 | ours better |
| b3_meta_init_only | wer | 250 | 15573.5 | 9.11e-01 | b3_meta_init_only better |
| b3_meta_init_only | utmos | 250 | 15578.0 | 9.24e-01 | b3_meta_init_only better |
| b3_v2 | simwavlm | 250 | 24.0 | 1.24e-42 | b3_v2 better |
| b3_v2 | simecapa | 250 | 0.0 | 9.25e-43 | b3_v2 better |
| b3_v2 | f0pcc | 219 | 9819.0 | 1.77e-02 | b3_v2 better |
| b3_v2 | wer | 250 | 9604.0 | 1.06e-07 | b3_v2 better |
| b3_v2 | utmos | 247 | 129.0 | 1.37e-41 | b3_v2 better |
| ours_v2 | simwavlm | 250 | 22.0 | 1.21e-42 | ours_v2 better |
| ours_v2 | simecapa | 250 | 0.0 | 9.25e-43 | ours_v2 better |
| ours_v2 | f0pcc | 221 | 9995.0 | 1.70e-02 | ours_v2 better |
| ours_v2 | wer | 250 | 8389.5 | 1.79e-10 | ours_v2 better |
| ours_v2 | utmos | 246 | 113.0 | 1.66e-41 | ours_v2 better |
