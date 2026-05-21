# EMNLP Baseline Failure Analysis

Date: 2026-05-19

## Current failure

The current `ours_emnlp` run does not beat vanilla F5-TTS on the primary
speaker-similarity metrics.

| method | split | SIM-o WavLM | ECAPA | ASR-Err | F0 PCC |
|---|---:|---:|---:|---:|---:|
| `b1_f5` | overall | 0.8887 | 0.5729 | 0.8188 | 0.1154 |
| `ours_emnlp` | overall | 0.8790 | 0.5655 | 0.8243 | 0.1378 |
| `b1_f5` | zero-shot | 0.8901 | 0.5658 | 0.9563 | 0.1035 |
| `ours_emnlp` | zero-shot | 0.8815 | 0.5644 | 0.9679 | 0.1118 |

The only consistent gain is F0, which is not enough for the paper claim and is
also a weak prompt-vs-generated-text metric. The main paper metric should be
speaker similarity, with ASR error and UTMOS as quality/intelligibility guards.

## Root causes

1. The learned loss weighter `phi` collapsed onto a disabled loss.

   `L_intel` is zero in training/eval-time TTT because ASR/CER is not
   differentiable here, and `inner_loss_lambda["intel"] = 0.0`. Before this
   fix, `phi` still produced a softmax over all five losses and the mask was
   applied only afterward. That let it minimize the inner objective by putting
   probability on the inactive zero slot.

   Probe on existing checkpoints:

   | checkpoint | raw `w_intel` | active weight mass before fix |
   |---|---:|---:|
   | `cycleadapt_v2/final.pt` | 0.7725 | 0.2275 |
   | `cycleadapt_emnlp_v3/final.pt` | 0.9122 | 0.0878 |

   This means the real adaptation gradient was shrunk by roughly 4x to 11x.
   After the patch, inactive losses get zero probability and active losses
   renormalize to 1.

2. The learned adapter stayed effectively identity.

   The LoRA `B` matrices in the trained checkpoints are near zero:

   | checkpoint | `B` std | `B` abs max |
   |---|---:|---:|
   | `cycleadapt_v2/final.pt` | 1.77e-7 | 3.64e-6 |
   | `cycleadapt_emnlp_v3/final.pt` | 1.85e-7 | 1.31e-5 |

   Since the LoRA correction depends on `A @ B`, the adapter is essentially a
   no-op. This matches the score pattern: we are paying TTT cost without a real
   correction.

3. The K=0 ablation was wrong.

   `scripts/08_method_ours_ttt.py` used `args.K_test or cfg.cycle.K_test`, so
   `--K-test 0` silently became the default K=3. The meta-init-only ablation was
   therefore not actually K=0.

4. The eval pipeline could reuse stale audio.

   Some generators skipped existing WAV files and some summaries did not carry
   generation metadata such as `final_nfe`, `final_cfg_strength`, or `K_test`.
   That made it too easy to compare new method scores against old audio.

5. The ASR metric was named too narrowly.

   The implementation computes WER for whitespace-tokenized languages and CER
   for CJK targets. The computation is reasonable, but the paper/table label
   should be `ASR-Err`, not plain `WER`.

## Fixes applied

- `src/cycle_tts/loss_weighter.py`
  - Added `active_mask`.
  - Masks inactive losses before softmax.
  - Floors/renormalizes only over active losses.

- `src/cycle_tts/losses.py`
  - `LossBundle.weighted_sum` now renormalizes after applying the loss mask.

- `src/cycle_tts/meta_trainer.py`
  - Passes the active mask into `phi`.
  - Adds a small residual gradient step (`residual_grad_lr`) to prevent `psi`
    from learning a pure no-op.
  - Logs average `w_spk`, `w_spec`, `w_f0`, `w_id`, and `w_intel`.

- `scripts/08_method_ours_ttt.py`
  - Honors `--K-test 0`.
  - Adds `--overwrite`.
  - Adds identity-aware final candidate selection with `--rerank-candidates`.
  - Records generation metadata in `summary.json`.

- `scripts/09_score_method.py`
  - Copies generation metadata into score summaries.
  - Adds a metric note explaining `wer` is ASR edit error: WER for
    whitespace-tokenized languages and CER for CJK.

- `scripts/run_emnlp_full.sh`
  - Regenerates and rescores B1 at the same NFE/CFG settings.
  - Runs adaptive methods with overwrite enabled.
  - Uses `--rerank-candidates 4` for our method and ablations.

- `scripts/12_aggregate_emnlp.py`
  - Relabels table column from `WER` to `ASR-Err`.

## Research context

F5-TTS is already a strong flow-matching baseline with DiT, ConvNeXt text
refinement, and Sway Sampling, so small post-hoc adapters must create a real
speaker-identity gain to beat it. Recent F5-style improvement work is
consistent with that diagnosis:

- F5-TTS: https://arxiv.org/abs/2410.06885
- F5R-TTS optimizes WER and SIM rewards directly with GRPO:
  https://arxiv.org/abs/2504.02407
- Cross-Lingual F5-TTS emphasizes prompt handling and duration/speaking-rate
  control for cross-lingual cloning: https://arxiv.org/abs/2509.14579

The immediate rescue strategy is therefore not to claim a broad modeling win
from the old run. It is to rerun with the fixed inner objective, verify that the
adapter actually moves, and use prompt-only identity selection as the cheap
test-time reward signal.

## Rerun

```bash
cd /home/ubuntu/CYCLE_TTS
bash scripts/run_emnlp_train_and_eval.sh
```

This writes the fixed checkpoint to:

```text
checkpoints/cycleadapt_emnlp_v3_fixed/final.pt
```

During training, watch `w_intel`. It should stay at or near `0.000` for active
TTT losses. Also watch the LoRA `B` statistics after training; if `B` remains at
1e-7 scale, the method is still not adapting.

