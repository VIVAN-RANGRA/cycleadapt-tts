#!/usr/bin/env python
"""Workshop-oriented analyses for the zh-expanded experiment.

This script does not generate audio.  It turns the completed per-item score
JSONL files into the extra evidence a workshop paper needs: paired confidence
intervals, failure buckets, and runtime/cost tables.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path("/home/ubuntu/CYCLE_TTS")
SCORES = ROOT / "results" / "scores"
AUDIO = ROOT / "results" / "audio"
TABLES = ROOT / "results" / "tables" / "workshop"
TABLES.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("b1_f5_zhx", "F5-TTS"),
    ("b1_f5_zhx_rerank8_asr", "F5 + verifier/ASR rerank"),
    ("cycleadapt_zhx_final", "CycleAdapt-Final"),
    ("cycleadapt_zhx_final_id", "Identity-only final"),
]
BASELINES = ["b1_f5_zhx", "b1_f5_zhx_rerank8_asr"]
FOCUS = ["cycleadapt_zhx_final", "cycleadapt_zhx_final_id"]
METRICS = ["simwavlm", "simecapa", "wer", "f0pcc", "utmos"]
LOWER_BETTER = {"wer"}


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_scores() -> Dict[str, List[Dict]]:
    return {method: load_jsonl(SCORES / f"{method}.jsonl") for method, _ in METHODS}


def finite(v) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(v)


def mean(rows: Iterable[Dict], metric: str) -> float:
    vals = [r.get(metric) for r in rows]
    vals = [v for v in vals if finite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def fmt(x: float) -> str:
    return "nan" if not finite(x) else f"{x:.4f}"


def paired_values(a_rows: List[Dict], b_rows: List[Dict], metric: str) -> List[Tuple[float, str]]:
    a = {r["item_id"]: r for r in a_rows}
    b = {r["item_id"]: r for r in b_rows}
    vals = []
    sign = -1.0 if metric in LOWER_BETTER else 1.0
    for item_id in sorted(set(a) & set(b)):
        av = a[item_id].get(metric)
        bv = b[item_id].get(metric)
        if finite(av) and finite(bv):
            vals.append((sign * (av - bv), item_id))
    return vals


def bootstrap_ci(vals: List[float], n_boot: int = 5000) -> Tuple[float, float, float]:
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(1337)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def significance(scores: Dict[str, List[Dict]]) -> None:
    rows = []
    labels = dict(METHODS)
    for method in FOCUS:
        for base in BASELINES:
            for metric in METRICS:
                vals_and_ids = paired_values(scores[method], scores[base], metric)
                vals = [v for v, _ in vals_and_ids]
                mu, lo, hi = bootstrap_ci(vals)
                rows.append({
                    "Method": labels[method],
                    "Baseline": labels[base],
                    "Metric": metric,
                    "n": len(vals),
                    "mean_benefit": fmt(mu),
                    "ci95_lo": fmt(lo),
                    "ci95_hi": fmt(hi),
                    "wins": sum(1 for v in vals if v > 0),
                    "win_rate": fmt(sum(1 for v in vals if v > 0) / len(vals)) if vals else "nan",
                    "note": "positive means method is better; WER is sign-flipped",
                })
    write_csv(TABLES / "paired_significance.csv", rows)


def bucket_ids(scores: Dict[str, List[Dict]]) -> Dict[str, set]:
    f5 = scores["b1_f5_zhx"]
    by_id = {r["item_id"]: r for r in f5}
    ids = set(by_id)
    sim_vals = np.asarray([r["simwavlm"] for r in f5 if finite(r.get("simwavlm"))])
    ecapa_vals = np.asarray([r["simecapa"] for r in f5 if finite(r.get("simecapa"))])
    wer_vals = np.asarray([r["wer"] for r in f5 if finite(r.get("wer"))])
    sim_q25 = float(np.quantile(sim_vals, 0.25))
    ecapa_q25 = float(np.quantile(ecapa_vals, 0.25))
    wer_q75 = float(np.quantile(wer_vals, 0.75))

    out = {
        "all": ids,
        "zero-shot": {i for i, r in by_id.items() if r["pair_class"] == "zero-shot"},
        "in-distrib": {i for i, r in by_id.items() if r["pair_class"] == "in-distrib"},
        "low_f5_wavlm_q25": {i for i, r in by_id.items() if finite(r.get("simwavlm")) and r["simwavlm"] <= sim_q25},
        "low_f5_ecapa_q25": {i for i, r in by_id.items() if finite(r.get("simecapa")) and r["simecapa"] <= ecapa_q25},
        "high_f5_asr_q75": {i for i, r in by_id.items() if finite(r.get("wer")) and r["wer"] >= wer_q75},
        "far_zh_hi_ja": {i for i, r in by_id.items() if r["pair_id"] in {"zh_hi", "zh_ja"}},
    }
    for pair in sorted({r["pair_id"] for r in f5}):
        out[pair] = {i for i, r in by_id.items() if r["pair_id"] == pair}
    return out


def failure_buckets(scores: Dict[str, List[Dict]]) -> None:
    labels = dict(METHODS)
    buckets = bucket_ids(scores)
    rows = []
    for bucket, ids in buckets.items():
        for method, label in METHODS:
            rs = [r for r in scores[method] if r["item_id"] in ids]
            row = {"Bucket": bucket, "Method": label, "n": len(rs)}
            for metric in METRICS:
                row[metric] = fmt(mean(rs, metric))
            rows.append(row)
    write_csv(TABLES / "failure_buckets.csv", rows)

    delta_rows = []
    id_rows = {m: {r["item_id"]: r for r in rows_} for m, rows_ in scores.items()}
    for bucket, ids in buckets.items():
        for method in FOCUS:
            for base in BASELINES:
                for metric in METRICS:
                    sign = -1.0 if metric in LOWER_BETTER else 1.0
                    vals = []
                    for item_id in sorted(ids):
                        a = id_rows[method].get(item_id, {}).get(metric)
                        b = id_rows[base].get(item_id, {}).get(metric)
                        if finite(a) and finite(b):
                            vals.append(sign * (a - b))
                    mu, lo, hi = bootstrap_ci(vals, n_boot=2000)
                    delta_rows.append({
                        "Bucket": bucket,
                        "Method": dict(METHODS)[method],
                        "Baseline": dict(METHODS)[base],
                        "Metric": metric,
                        "n": len(vals),
                        "mean_benefit": fmt(mu),
                        "ci95_lo": fmt(lo),
                        "ci95_hi": fmt(hi),
                        "wins": sum(1 for v in vals if v > 0),
                    })
    write_csv(TABLES / "bucket_deltas.csv", delta_rows)


def runtime_table() -> None:
    rows = []
    labels = dict(METHODS)
    for method, label in METHODS:
        p = AUDIO / method / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        n = s.get("n_generated") or s.get("n_items") or 0
        wall = float(s.get("wall_seconds", float("nan")))
        rows.append({
            "Method": label,
            "n_generated": n,
            "wall_seconds": fmt(wall),
            "seconds_per_item": fmt(wall / n) if n else "nan",
            "mean_rtf": fmt(float(s.get("mean_rtf", float("nan")))),
            "K_test": s.get("K_test", ""),
            "rerank_candidates": s.get("rerank_candidates", ""),
            "rerank_asr_topk": s.get("rerank_asr_topk", ""),
            "rerank_asr_model": s.get("rerank_asr_model_size", ""),
        })
    write_csv(TABLES / "runtime.csv", rows)


def write_markdown_note(scores: Dict[str, List[Dict]]) -> None:
    sig_path = TABLES / "paired_significance.csv"
    bucket_path = TABLES / "failure_buckets.csv"
    runtime_path = TABLES / "runtime.csv"
    lines = [
        "# Workshop Analysis Summary",
        "",
        "Generated from the completed zh-expanded experiment.",
        "",
        "Files:",
        f"- `{sig_path}`: paired bootstrap confidence intervals and win rates.",
        f"- `{bucket_path}`: metrics by failure bucket and language pair.",
        f"- `{TABLES / 'bucket_deltas.csv'}`: bucket-level paired benefits.",
        f"- `{runtime_path}`: runtime/cost table from generation summaries.",
        "",
        "Primary reading guide:",
        "- Treat positive `mean_benefit` as better; ASR error is sign-flipped.",
        "- Use Identity-only final as the main stable CycleAdapt variant.",
        "- Use failure buckets to support where adaptation helps, rather than overclaiming average SOTA.",
        "",
    ]
    (TABLES / "workshop_summary.md").write_text("\n".join(lines))


def main() -> None:
    scores = load_scores()
    missing = [m for m, rows in scores.items() if not rows]
    if missing:
        raise SystemExit(f"Missing score files for: {missing}")
    significance(scores)
    failure_buckets(scores)
    runtime_table()
    write_markdown_note(scores)
    print(f"Wrote workshop analysis tables -> {TABLES}")


if __name__ == "__main__":
    main()
