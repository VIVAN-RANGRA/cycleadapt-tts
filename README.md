# CycleAdapt-TTS

Prompt-only test-time speaker adaptation for cross-lingual F5-TTS voice cloning.

This repository contains the code, scripts, compact result tables, and final
paper-writing materials from the CycleAdapt-TTS experiment series.  Large
datasets, generated audio, checkpoints, logs, and local caches are intentionally
excluded from git.

## Current Scientific Takeaway

CycleAdapt is **not** a clean replacement for verifier reranking on easy clean
prompts.  The useful finding is sharper:

> Test-time adaptation helps most when the speaker prompt is fragile.  Under
> short or noisy prompt conditions, CycleAdapt improves speaker preservation
> substantially over vanilla F5-TTS; under noisy zero-shot transfer,
> CycleAdapt-Final also improves ASR error.

This is best framed as a strong workshop-style paper about benefits, tradeoffs,
and failure modes, not as a broad SOTA claim.

## Key Results

Clean Chinese-source zero-shot transfer:

- Vanilla F5-TTS: SIM-o `0.890`, ASR-Err `1.258`.
- F5 + verifier/ASR rerank: SIM-o `0.891`, ASR-Err `1.177`.
- CycleAdapt-Final: SIM-o `0.894`, ASR-Err `1.324`.
- Identity-only final: SIM-o `0.895`, ASR-Err `1.222`.

Stress tests are stronger:

- Short 3s prompt, zero-shot SIM-o:
  - F5-TTS `0.722`
  - F5 + rerank `0.782`
  - CycleAdapt-Final `0.783`
  - Identity-only `0.785`
- Noisy 10 dB prompt, zero-shot:
  - F5-TTS: SIM-o `0.659`, ASR-Err `1.131`
  - F5 + rerank: SIM-o `0.792`, ASR-Err `1.274`
  - CycleAdapt-Final: SIM-o `0.799`, ASR-Err `1.051`

The paper-ready results live in [`docs/final`](docs/final).

## Repository Layout

- `src/cycle_tts/`: core CycleAdapt modules.
- `scripts/`: training, evaluation, scoring, aggregation, and status scripts.
- `configs/`: local config stubs.
- `docs/`: analysis notes and final paper materials.
- `results/tables/`: compact CSV/Markdown result tables.
- `results/scores/`: compact per-item score JSONL and summary JSON files.
- `third_party_f5/`: expected F5-TTS checkout/submodule target.

Excluded from git:

- `data/`
- `checkpoints/`
- `logs/`
- `results/audio/`
- `results/prompts/`
- `.venv/`

## Quick Start

```bash
cd /home/ubuntu/CYCLE_TTS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Clone or initialize the F5-TTS dependency at `third_party_f5`:

```bash
git clone https://github.com/SWivid/F5-TTS.git third_party_f5
```

Then source the local environment:

```bash
source scripts/env.sh
```

## Data

The project expects:

- LibriTTS-R for English training episodes.
- AISHELL-3 for Chinese training episodes.
- FLEURS for multilingual held-out prompts/texts.
- VCTK for English held-out prompt speakers.

Use:

```bash
bash scripts/01a_download_data.sh
python scripts/02_prepare_vctk_fleurs.py
python scripts/03_build_manifests.py
```

Some environments use Hugging Face cache paths and pre-extracted FLEURS files;
see [`JUMPOFF.md`](JUMPOFF.md) for the detailed expected layout.

## Main Reproduction Scripts

Clean zh-expanded benchmark:

```bash
bash scripts/run_zh_expanded_experiment.sh
```

Final-only CycleAdapt rerun:

```bash
bash scripts/run_zh_expanded_final_only.sh
```

Workshop bundle:

```bash
nohup bash scripts/run_workshop_bundle.sh > logs/runs/workshop_bundle.log 2>&1 &
```

Status helpers:

```bash
scripts/status_zh_expanded.sh
scripts/status_workshop_bundle.sh
```

## Paper Materials

Run:

```bash
python scripts/17_make_final_materials.py
```

This creates:

- `docs/final/RESULTS_BRIEF.md`
- `docs/final/EXPERIMENT_INDEX.md`
- `docs/final/PAPER_OUTLINE.md`
- `docs/final/tables/`
- `docs/final/figures/`

Start writing from [`docs/final/README.md`](docs/final/README.md).
