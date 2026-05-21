#!/usr/bin/env python
"""Create paper-ready Markdown tables and figures under docs/final."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/CYCLE_TTS")
FINAL = ROOT / "docs" / "final"
TABLES = FINAL / "tables"
FIGS = FINAL / "figures"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_md_table(rows: list[dict], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n")
        return
    fields = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |"]
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def copy_and_markdown(src: Path, name: str, title: str) -> list[dict]:
    rows = read_csv(src)
    shutil.copy2(src, TABLES / f"{name}.csv")
    write_md_table(rows, TABLES / f"{name}.md", title)
    return rows


def as_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def plot_clean_main(rows: list[dict]) -> None:
    methods = []
    sim = []
    asr = []
    for r in rows:
        if r["Setting"] == "zero-shot":
            methods.append(r["Method"])
            sim.append(as_float(r["SIM-o"]))
            asr.append(as_float(r["ASR-Err"]))
    x = range(len(methods))
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar([i - 0.18 for i in x], sim, width=0.36, label="SIM-o", color="#377eb8")
    ax1.set_ylabel("SIM-o (higher better)")
    ax1.set_ylim(0.86, max(sim) + 0.015)
    ax2 = ax1.twinx()
    ax2.bar([i + 0.18 for i in x], asr, width=0.36, label="ASR-Err", color="#e41a1c")
    ax2.set_ylabel("ASR-Err (lower better)")
    ax2.set_ylim(0.9, max(asr) + 0.15)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_title("Clean Chinese-source zero-shot transfer")
    fig.tight_layout()
    fig.savefig(FIGS / "clean_zero_shot_sim_asr.png", dpi=180)
    plt.close(fig)


def plot_stress(rows: list[dict]) -> None:
    for cond in ["short3", "noise10"]:
        sub = [r for r in rows if r["Condition"] == cond and r["Setting"] == "zero-shot"]
        methods = [r["Method"] for r in sub]
        sim = [as_float(r["SIM-o"]) for r in sub]
        asr = [as_float(r["ASR-Err"]) for r in sub]
        x = range(len(methods))
        fig, ax1 = plt.subplots(figsize=(10, 4.8))
        ax1.bar([i - 0.18 for i in x], sim, width=0.36, label="SIM-o", color="#4daf4a")
        ax1.set_ylabel("SIM-o (higher better)")
        ax1.set_ylim(min(sim) - 0.03, max(sim) + 0.03)
        ax2 = ax1.twinx()
        ax2.bar([i + 0.18 for i in x], asr, width=0.36, label="ASR-Err", color="#984ea3")
        ax2.set_ylabel("ASR-Err (lower better)")
        ax2.set_ylim(min(asr) - 0.08, max(asr) + 0.15)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(methods, rotation=20, ha="right")
        ax1.set_title(f"Stress condition: {cond}, zero-shot")
        fig.tight_layout()
        fig.savefig(FIGS / f"stress_{cond}_zero_shot_sim_asr.png", dpi=180)
        plt.close(fig)


def plot_stress_deltas(rows: list[dict]) -> None:
    sim_rows = [
        r for r in rows
        if r["Metric"] == "simwavlm" and r["Baseline"] == "F5-TTS"
    ]
    labels = [f"{r['Condition']}\n{r['Method'].replace(' final', '')}" for r in sim_rows]
    vals = [as_float(r["mean_benefit"]) for r in sim_rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(vals)), vals, color="#ff7f00")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("SIM-o benefit over F5-TTS")
    ax.set_title("Speaker-similarity gains under prompt stress")
    fig.tight_layout()
    fig.savefig(FIGS / "stress_sim_benefit_over_f5.png", dpi=180)
    plt.close(fig)


def plot_runtime(rows: list[dict]) -> None:
    methods = [r["Method"] for r in rows]
    vals = [as_float(r["seconds_per_item"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(range(len(vals)), vals, color="#a65628")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Seconds per item")
    ax.set_title("Inference cost on zh-expanded evaluation")
    fig.tight_layout()
    fig.savefig(FIGS / "runtime_seconds_per_item.png", dpi=180)
    plt.close(fig)


def write_final_docs(clean_rows: list[dict], stress_rows: list[dict], delta_rows: list[dict], sig_rows: list[dict]) -> None:
    readme = """# Final Paper Materials

This folder is the compact source of truth for writing the workshop paper.
It contains the tables, plots, and interpretation distilled from the completed
CycleAdapt-TTS experiments.  Large artifacts such as datasets, generated audio,
checkpoints, and logs are intentionally not included in git.

## Strongest Takeaway

CycleAdapt is not a clean SOTA replacement for verifier reranking on clean
prompts.  Its stronger contribution is robustness: under short or noisy prompt
conditions, adaptation substantially improves speaker preservation over vanilla
F5-TTS, and under noisy zero-shot transfer CycleAdapt-Final improves both
speaker similarity and ASR error relative to vanilla F5 and the reranked F5
baseline.

## Recommended Paper Claim

> Prompt-only test-time adaptation is useful for fragile cross-lingual voice
> cloning conditions.  Identity-only and CycleAdapt variants improve speaker
> preservation under short/noisy prompts, while intelligibility remains the key
> tradeoff and requires explicit ASR-aware treatment.

## Contents

- `RESULTS_BRIEF.md`: short narrative for the results section.
- `EXPERIMENT_INDEX.md`: what was run and where outputs live.
- `PAPER_OUTLINE.md`: workshop paper skeleton.
- `tables/`: Markdown and CSV copies of paper-ready tables.
- `figures/`: PNG plots for the paper draft.
"""
    (FINAL / "README.md").write_text(readme)

    brief = """# Results Brief

## Clean Chinese-Source Evaluation

The clean zh-expanded benchmark has 175 items across `zh->en`, `zh->zh`, and
zero-shot `zh->{es,fr,de,hi,ja}`.  All methods finished with `175/175` outputs.

Clean average results are modest: CycleAdapt improves speaker similarity over
vanilla F5-TTS, but does not decisively dominate the strong F5 + verifier/ASR
reranking baseline.  This should be framed honestly.

Key clean zero-shot numbers:

- F5-TTS: SIM-o 0.890, ASR-Err 1.258, UTMOS 1.541.
- F5 + verifier/ASR rerank: SIM-o 0.891, ASR-Err 1.177, UTMOS 1.517.
- CycleAdapt-Final: SIM-o 0.894, ASR-Err 1.324, UTMOS 1.538.
- Identity-only final: SIM-o 0.895, ASR-Err 1.222, UTMOS 1.546.

## Stress Tests

The strongest workshop evidence comes from prompt stress tests using a balanced
70-item subset per condition, 10 items per language pair.

Short 3-second prompts:

- F5-TTS zero-shot SIM-o: 0.722.
- F5 + rerank zero-shot SIM-o: 0.782.
- CycleAdapt-Final zero-shot SIM-o: 0.783.
- Identity-only final zero-shot SIM-o: 0.785.

Noisy 10 dB prompts:

- F5-TTS zero-shot SIM-o: 0.659, ASR-Err 1.131.
- F5 + rerank zero-shot SIM-o: 0.792, ASR-Err 1.274.
- CycleAdapt-Final zero-shot SIM-o: 0.799, ASR-Err 1.051.
- Identity-only final zero-shot SIM-o: 0.793, ASR-Err 1.182.

This gives a clean workshop claim: CycleAdapt is most useful when the prompt is
fragile.  In the noisy zero-shot condition, CycleAdapt-Final improves both
speaker preservation and ASR error over vanilla F5 and the reranked baseline.

## Caveat

The method is expensive.  Clean zh-expanded generation costs about 21-22 seconds
per item for CycleAdapt/reranked methods versus about 1.3 seconds for vanilla
F5-TTS.  The paper should state this plainly.
"""
    (FINAL / "RESULTS_BRIEF.md").write_text(brief)

    index = """# Experiment Index

## Main Clean zh-expanded Experiment

Purpose: Chinese-source cross-lingual evaluation on clean prompts.

Eval set:
- Built by `scripts/06_build_eval_set_zh_expanded.py`.
- 175 total items, 25 per language pair.
- Pairs: `zh_en`, `zh_zh`, `zh_es`, `zh_fr`, `zh_de`, `zh_hi`, `zh_ja`.

Methods:
- `b1_f5_zhx`: vanilla F5-TTS.
- `b1_f5_zhx_rerank8_asr`: F5 with verifier/ASR reranking.
- `cycleadapt_zhx_final`: CycleAdapt-Final.
- `cycleadapt_zhx_final_id`: identity-only adaptation.

Key scripts:
- `scripts/run_zh_expanded_experiment.sh`: full baseline + final run.
- `scripts/run_zh_expanded_final_only.sh`: re-run only CycleAdapt final methods.
- `scripts/13_aggregate_zh_expanded.py`: aggregate tables.

Important tables:
- `results/tables/zh_expanded/zh_table_main.csv`
- `results/tables/zh_expanded/zh_table_by_pair.csv`
- `results/tables/zh_expanded/zh_table_deltas.csv`

## Workshop Analysis

Purpose: paired significance, failure buckets, runtime/cost.

Scripts:
- `scripts/14_workshop_analysis.py`

Tables:
- `results/tables/workshop/paired_significance.csv`
- `results/tables/workshop/failure_buckets.csv`
- `results/tables/workshop/bucket_deltas.csv`
- `results/tables/workshop/runtime.csv`

## Prompt Stress Tests

Purpose: evaluate short/noisy prompt robustness.

Eval sets:
- Built by `scripts/14_build_workshop_stress_eval.py`.
- `short3`: prompt truncated to 3 seconds.
- `noise10`: prompt corrupted with 10 dB Gaussian noise.
- 70 items per condition, balanced across seven language pairs.

Runner:
- `scripts/run_workshop_bundle.sh`

Aggregator:
- `scripts/16_aggregate_workshop_stress.py`

Tables:
- `results/tables/workshop_stress/stress_main.csv`
- `results/tables/workshop_stress/stress_by_pair.csv`
- `results/tables/workshop_stress/stress_deltas.csv`
"""
    (FINAL / "EXPERIMENT_INDEX.md").write_text(index)

    outline = """# Workshop Paper Outline

## Title

Test-Time Speaker Adaptation for Cross-Lingual Voice Cloning: Benefits,
Tradeoffs, and Failure Modes

## Abstract

Emphasize prompt-only adaptation, cross-lingual F5-TTS, and robustness under
fragile prompt conditions.  Avoid claiming broad SOTA.

## 1. Introduction

- Cross-lingual voice cloning depends heavily on prompt quality.
- Modern F5-TTS is already strong under clean prompts.
- Test-time adaptation may help when prompt evidence is fragile.
- Main finding: average clean gains are modest, but stress gains are substantial.

## 2. Method

- Frozen F5-TTS backbone.
- Identity Alignment Adapter (LoRA).
- Learned/test-time updates on prompt-only losses.
- Variants: full CycleAdapt and identity-only adaptation.
- Verifier reranking applied consistently.

## 3. Experimental Setup

- Clean zh-expanded benchmark: 175 items, seven Chinese-source language pairs.
- Stress benchmark: balanced 70-item subsets for short and noisy prompt conditions.
- Metrics: WavLM SIM-o, ECAPA SIM, ASR-Err, UTMOS, F0 PCC.
- Baselines: F5-TTS and F5 + verifier/ASR reranking.

## 4. Results

- Clean setting: modest speaker gains, ASR tradeoff.
- Short prompt: Identity-only achieves best zero-shot SIM-o.
- Noisy prompt: CycleAdapt-Final achieves best zero-shot SIM-o and ASR-Err.
- Failure buckets: gains concentrate in low baseline-similarity cases.
- Runtime: adaptation is expensive.

## 5. Discussion

- Test-time adaptation helps when the prompt is unreliable.
- Reranking is a very strong baseline; adaptation should be framed as robustifying
  candidate generation, not replacing reranking.
- Intelligibility remains unresolved; future work should add differentiable or
  proxy ASR objectives inside adaptation.

## 6. Limitations

- Automatic metrics only, no listening panel.
- Stress tests use 70-item balanced subsets rather than full 175.
- CPU tiny ASR rerank is a pragmatic stability choice.
- Inference cost is high.
"""
    (FINAL / "PAPER_OUTLINE.md").write_text(outline)

    figs = """# Figures

- `clean_zero_shot_sim_asr.png`: clean zh-expanded zero-shot SIM-o and ASR-Err.
- `stress_short3_zero_shot_sim_asr.png`: short-prompt zero-shot robustness.
- `stress_noise10_zero_shot_sim_asr.png`: noisy-prompt zero-shot robustness.
- `stress_sim_benefit_over_f5.png`: speaker-similarity benefit over vanilla F5.
- `runtime_seconds_per_item.png`: inference cost.
"""
    (FIGS / "README.md").write_text(figs)


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    clean = copy_and_markdown(ROOT / "results/tables/zh_expanded/zh_table_main.csv", "zh_clean_main", "Clean zh-expanded main table")
    copy_and_markdown(ROOT / "results/tables/zh_expanded/zh_table_by_pair.csv", "zh_clean_by_pair", "Clean zh-expanded by language pair")
    copy_and_markdown(ROOT / "results/tables/zh_expanded/zh_table_deltas.csv", "zh_clean_deltas", "Clean zh-expanded paired deltas")
    stress = copy_and_markdown(ROOT / "results/tables/workshop_stress/stress_main.csv", "stress_main", "Prompt stress main table")
    stress_delta = copy_and_markdown(ROOT / "results/tables/workshop_stress/stress_deltas.csv", "stress_deltas", "Prompt stress paired deltas")
    copy_and_markdown(ROOT / "results/tables/workshop_stress/stress_by_pair.csv", "stress_by_pair", "Prompt stress by language pair")
    sig = copy_and_markdown(ROOT / "results/tables/workshop/paired_significance.csv", "paired_significance", "Paired bootstrap significance")
    copy_and_markdown(ROOT / "results/tables/workshop/failure_buckets.csv", "failure_buckets", "Failure buckets")
    copy_and_markdown(ROOT / "results/tables/workshop/bucket_deltas.csv", "bucket_deltas", "Failure-bucket paired deltas")
    runtime = copy_and_markdown(ROOT / "results/tables/workshop/runtime.csv", "runtime", "Runtime and inference cost")

    plot_clean_main(clean)
    plot_stress(stress)
    plot_stress_deltas(stress_delta)
    plot_runtime(runtime)
    write_final_docs(clean, stress, stress_delta, sig)
    print(f"Wrote final paper materials -> {FINAL}")


if __name__ == "__main__":
    main()
