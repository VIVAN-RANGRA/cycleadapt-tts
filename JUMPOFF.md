# JUMPOFF: CycleAdapt-TTS Codebase and Experiment Continuation Guide

This file is for the next coding/research agent.  It explains the repository,
what was tried, what worked, what did not, and where to continue.

## 1. High-Level Project

CycleAdapt-TTS studies prompt-only test-time adaptation for cross-lingual
voice cloning with F5-TTS.  The system keeps the F5-TTS backbone frozen and
adds a lightweight LoRA-style Identity Alignment Adapter (IAA).  At evaluation
time, it adapts on the prompt only, then generates target-language speech.

Core question:

> Can prompt-only test-time adaptation improve speaker preservation in
> cross-lingual TTS, especially when prompts are short/noisy/fragile?

Current answer:

- Clean prompts: gains over vanilla F5 are modest, and reranking is a very
  strong baseline.
- Fragile prompts: CycleAdapt gives a much stronger speaker-similarity gain.
- Intelligibility/ASR remains the key tradeoff.

## 2. Important Directories

- `src/cycle_tts/`: method implementation.
- `scripts/`: all train/eval/scoring/aggregation runners.
- `docs/final/`: paper-ready Markdown, tables, and figures.
- `docs/results/`: narrative result discussion and paper-claim guidance.
- `results/tables/`: compact result CSVs.
- `results/scores/`: score JSONLs and summary JSONs.
- `third_party_f5/`: F5-TTS checkout target.

Not tracked:

- `data/`: datasets and caches.
- intermediate checkpoint steps: `checkpoints/*/step*.pt`.
- `logs/`: nohup/run logs.
- `results/audio/`: generated wavs.
- `results/prompts/`: derived short/noisy prompt wavs.
- `.venv/`: Python environment.

Tracked for reproducibility:

- `checkpoints/*/final.pt`: final CycleAdapt adapter/model-state checkpoints.
- `results/eval_set*.jsonl`: exact clean and stress eval splits.
- `results/scores/`: per-item and summary metrics.
- `results/tables/`: aggregate paper tables.

## 3. Data Needed

Training/eval uses:

- LibriTTS-R: English multi-speaker training.
- AISHELL-3: Chinese multi-speaker training.
- FLEURS: multilingual held-out prompts and target text pools.
- VCTK: English held-out prompt speakers.

Expected local layout is under:

- `data/manifests/`
- `data/fleurs_extracted/`
- `data/cache/huggingface/`
- `data/cache/torch/`

Primary setup scripts:

- `scripts/01a_download_hf.py`
- `scripts/01a_download_data.sh`
- `scripts/02_prepare_vctk_fleurs.py`
- `scripts/03_build_manifests.py`

Recommended setup on a fresh GPU box:

```bash
cd /home/ubuntu/CYCLE_TTS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git submodule update --init --recursive
source scripts/env.sh

# Fast HF/CDN path used by the current workspace.
python scripts/01a_download_hf.py --datasets vctk aishell3 libritts_r fleurs
python scripts/02_prepare_vctk_fleurs.py
python scripts/03_build_manifests.py
```

Alternative direct-download path:

```bash
bash scripts/01a_download_data.sh
python scripts/03_build_manifests.py
```

The HF path creates/parses:

- `data/vctk_raw/` and `data/vctk_extracted/`
- `data/aishell3_raw/`
- `data/libritts_r/`
- `data/fleurs/` and `data/fleurs_extracted/`
- `data/cache/huggingface/`

The direct path creates/parses:

- `data/vctk/`
- `data/aishell3/`
- `data/libritts_r/`

Do not commit these directories.  They are large and machine-local.

After data prep, expected manifest files include:

- `data/manifests/libritts_r_train.jsonl`
- `data/manifests/aishell3_train.jsonl`
- `data/manifests/vctk_eval.jsonl`
- `data/manifests/fleurs_eval.jsonl`

The exact paper eval-set JSONLs are tracked and can be used directly once the
referenced audio paths exist:

- `results/eval_set.jsonl`
- `results/eval_set_zh_expanded.jsonl`
- `results/eval_set_zh_workshop_short3.jsonl`
- `results/eval_set_zh_workshop_noise10.jsonl`

The expanded Chinese-source eval requires FLEURS languages:

- `zh`, `en`, `es`, `fr`, `de`, `hi`, `ja`

Korean was considered but not used because the local FLEURS manifest did not
have `ko`.

## 4. Main Method Files

- `src/cycle_tts/iaa.py`
  - LoRA-style Identity Alignment Adapter around selected F5 transformer
    modules.

- `src/cycle_tts/meta_optimizer.py`
  - Learned optimizer `psi`.

- `src/cycle_tts/loss_weighter.py`
  - Learned loss weighter `phi`.
  - Important fix: inactive losses must be masked before softmax.  Earlier
    `phi` collapsed onto inactive/intelligibility slots.

- `src/cycle_tts/losses.py`
  - Speaker, spectral, F0, identity, and placeholder intelligibility loss
    bundle.

- `src/cycle_tts/meta_trainer.py`
  - Meta-training loop.

- `src/cycle_tts/metrics.py`
  - WavLM SIM, ECAPA SIM, ASR-Err, UTMOS, F0 metrics.
  - ASR-Err is WER for whitespace-tokenized languages and CER for CJK.

- `scripts/08_method_ours_ttt.py`
  - Main test-time adaptation/generation script.
  - Supports `--id-only-ttt`, `--no-phi`, reranking, ASR top-k rerank.

## 5. Major Bugs Fixed

1. `phi` collapsed onto inactive zero losses.
   - Fixed by active-mask softmax.

2. Adapter collapse / unsafe LoRA scaling.
   - Added stability clamps and residual update control.

3. `--K-test 0` was not respected.
   - `args.K_test or default` was wrong because `0` is falsy.

4. Eval could silently reuse stale audio.
   - Added overwrite/fail-fast behavior and clearer summaries.

5. ASR-aware reranking OOMed on GPU.
   - The bad setup used faster-whisper on the same GPU as F5/WavLM/ECAPA for
     all 8 candidates.
   - Final stable setup: WavLM+ECAPA score all 8 candidates, then ASR-check only
     top 2 with CPU tiny/int8.

## 6. Completed Experiments

### 6.1 Clean zh-expanded Evaluation

Eval builder:

```bash
python scripts/06_build_eval_set_zh_expanded.py
```

Pairs:

- `zh_en`
- `zh_zh`
- `zh_es`
- `zh_fr`
- `zh_de`
- `zh_hi`
- `zh_ja`

Methods:

- `b1_f5_zhx`: vanilla F5.
- `b1_f5_zhx_rerank8_asr`: F5 + verifier/ASR rerank.
- `cycleadapt_zhx_final`: full CycleAdapt final.
- `cycleadapt_zhx_final_id`: identity-only final.

Main tables:

- `results/tables/zh_expanded/zh_table_main.csv`
- `results/tables/zh_expanded/zh_table_by_pair.csv`
- `results/tables/zh_expanded/zh_table_deltas.csv`

Clean zero-shot summary:

- F5-TTS: SIM-o `0.890`, ASR-Err `1.258`, UTMOS `1.541`.
- F5 + rerank: SIM-o `0.891`, ASR-Err `1.177`, UTMOS `1.517`.
- CycleAdapt-Final: SIM-o `0.894`, ASR-Err `1.324`, UTMOS `1.538`.
- Identity-only: SIM-o `0.895`, ASR-Err `1.222`, UTMOS `1.546`.

Interpretation:

- Speaker similarity improves modestly over vanilla F5.
- Reranked F5 remains a strong baseline.
- ASR/intelligibility is not solved by CycleAdapt.

### 6.2 Workshop Analysis

Script:

```bash
python scripts/14_workshop_analysis.py
```

Outputs:

- `results/tables/workshop/paired_significance.csv`
- `results/tables/workshop/failure_buckets.csv`
- `results/tables/workshop/bucket_deltas.csv`
- `results/tables/workshop/runtime.csv`

Useful findings:

- Identity-only has a small but positive clean SIM-o benefit over F5.
- Gains are clearer in low baseline-SIM buckets.
- Runtime is expensive: about 21-22 sec/item for adaptation/rerank versus about
  1.3 sec/item for vanilla F5.

### 6.3 Workshop Stress Tests

Runner:

```bash
bash scripts/run_workshop_bundle.sh
```

Conditions:

- `short3`: prompt truncated to 3 seconds.
- `noise10`: prompt corrupted with 10 dB Gaussian noise.

Each condition uses 70 items:

- 7 language pairs.
- 10 items per pair.

Outputs:

- `results/tables/workshop_stress/stress_main.csv`
- `results/tables/workshop_stress/stress_by_pair.csv`
- `results/tables/workshop_stress/stress_deltas.csv`

Strongest results:

Short 3s zero-shot SIM-o:

- F5-TTS: `0.722`
- F5 + rerank: `0.782`
- CycleAdapt-Final: `0.783`
- Identity-only final: `0.785`

Noisy 10 dB zero-shot:

- F5-TTS: SIM-o `0.659`, ASR-Err `1.131`
- F5 + rerank: SIM-o `0.792`, ASR-Err `1.274`
- CycleAdapt-Final: SIM-o `0.799`, ASR-Err `1.051`
- Identity-only final: SIM-o `0.793`, ASR-Err `1.182`

This is the best workshop claim: CycleAdapt is useful when prompt quality is
fragile.

### 6.4 Where The Model Weights Are

Final learned weights are tracked:

- `checkpoints/cycleadapt_v1/final.pt`
- `checkpoints/cycleadapt_v2/final.pt`
- `checkpoints/cycleadapt_emnlp_v3/final.pt`
- `checkpoints/cycleadapt_emnlp_v3_fixed/final.pt`
- `checkpoints/proto_B4M5/final.pt`
- `checkpoints/proto_M3/final.pt`
- `checkpoints/proto_speed/final.pt`
- `checkpoints/proto_speed2/final.pt`

The paper runs mainly use:

```bash
checkpoints/cycleadapt_emnlp_v3_fixed/final.pt
```

The v3 fixed run warm-started from:

```bash
checkpoints/cycleadapt_v2/final.pt
```

Intermediate `step*.pt` checkpoints remain untracked because they are not needed
to reproduce evaluation and mostly duplicate the final checkpoint state.

## 7. What Kind of Paper?

Current best target: strong workshop paper.

Do not claim broad SOTA.  The honest story is:

> Test-time speaker adaptation provides modest clean-prompt gains but becomes
> more valuable under short/noisy prompt conditions.  It improves speaker
> preservation and exposes an intelligibility tradeoff that future ASR-aware
> adaptation should address.

Suggested title:

> Test-Time Speaker Adaptation for Cross-Lingual Voice Cloning: Benefits,
> Tradeoffs, and Failure Modes

## 8. How To Continue

Highest-value next steps:

1. Write the workshop paper from `docs/final/`.
2. Add ASR-aware/intelligibility proxy into adaptation, not just reranking.
3. Run full 175-item stress tests only if a stronger main-conference claim is
   needed.
4. Meta-train on multilingual target episodes if aiming for Findings/long paper.

Do not spend first effort on more random averages.  The result is already clear:
adaptation helps under fragile prompts; intelligibility is the bottleneck.

Also read `docs/results/` before writing.  It contains the compact argument for
clean results, stress results, failure buckets, limitations, and artifact usage.

## 9. Command Cheat Sheet

Environment:

```bash
cd /home/ubuntu/CYCLE_TTS
source scripts/env.sh
```

Clean zh-expanded:

```bash
bash scripts/run_zh_expanded_experiment.sh
```

Final-only rerun:

```bash
bash scripts/run_zh_expanded_final_only.sh
```

Workshop bundle:

```bash
nohup bash scripts/run_workshop_bundle.sh > logs/runs/workshop_bundle.log 2>&1 &
```

Status:

```bash
scripts/status_zh_expanded.sh
scripts/status_workshop_bundle.sh
```

Final materials:

```bash
python scripts/17_make_final_materials.py
```

## 10. Do Not Commit

Do not commit:

- `data/`
- `checkpoints/*/step*.pt`
- `logs/`
- `results/audio/`
- `results/prompts/`
- `.venv/`
- Hugging Face caches
- generated wavs

The repo should contain code, scripts, final checkpoints, eval-set JSONLs,
compact score/table artifacts, and paper-ready docs only.
