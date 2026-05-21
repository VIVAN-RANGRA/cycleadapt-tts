#!/usr/bin/env python
"""Aggregate expanded Chinese-source results."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/CYCLE_TTS")
SCORES = ROOT / "results" / "scores"
EVAL_SET = ROOT / "results" / "eval_set_zh_expanded.jsonl"
TABLES = ROOT / "results" / "tables" / "zh_expanded"
TABLES.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("b1_f5_zhx", "F5-TTS"),
    ("b1_f5_zhx_rerank8_asr", "F5 + verifier/ASR rerank"),
    ("cycleadapt_zhx_final", "CycleAdapt-Final"),
    ("cycleadapt_zhx_final_id", "Identity-only final"),
]
OURS = "cycleadapt_zhx_final"
METRICS = ["simwavlm", "simecapa", "wer", "f0pcc", "utmos"]


def load_summary(method: str):
    p = SCORES / f"{method}.summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_items(method: str):
    p = SCORES / f"{method}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def mean(rows, metric):
    vals = [r.get(metric) for r in rows]
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def fmt(x):
    return "---" if not isinstance(x, (int, float)) or math.isnan(x) else f"{x:.3f}"


def main() -> None:
    expected_n = sum(1 for line in EVAL_SET.read_text().splitlines() if line.strip())
    incomplete = []
    for key, _ in METHODS:
        s = load_summary(key)
        if s and s.get("n_available") != expected_n:
            incomplete.append(f"{key}: {s.get('n_available')}/{expected_n}")
    if incomplete:
        raise SystemExit("Refusing to aggregate incomplete zh-expanded scores: " + ", ".join(incomplete))

    # Main class-level table.
    rows = []
    for key, label in METHODS:
        s = load_summary(key)
        if not s:
            continue
        for split in ["in-distrib", "zero-shot"]:
            d = s.get("by_class", {}).get(split, {})
            if not d:
                continue
            rows.append({
                "Method": label,
                "Setting": split,
                "SIM-o": fmt(d["simwavlm"]["mean"]),
                "ECAPA": fmt(d["simecapa"]["mean"]),
                "ASR-Err": fmt(d["wer"]["mean"]),
                "F0": fmt(d["f0pcc"]["mean"]),
                "UTMOS": fmt(d["utmos"]["mean"]),
            })
    with (TABLES / "zh_table_main.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    # Per-pair table.
    pair_rows = []
    items_by_method = {k: load_items(k) for k, _ in METHODS}
    pairs = sorted({r["pair_id"] for rows_ in items_by_method.values() for r in rows_ if r["pair_id"].startswith("zh_")})
    for pair in pairs:
        for key, label in METHODS:
            rs = [r for r in items_by_method.get(key, []) if r["pair_id"] == pair]
            if not rs:
                continue
            pair_rows.append({
                "Pair": pair,
                "Method": label,
                "n": len(rs),
                "SIM-o": fmt(mean(rs, "simwavlm")),
                "ECAPA": fmt(mean(rs, "simecapa")),
                "ASR-Err": fmt(mean(rs, "wer")),
                "F0": fmt(mean(rs, "f0pcc")),
                "UTMOS": fmt(mean(rs, "utmos")),
            })
    with (TABLES / "zh_table_by_pair.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()) if pair_rows else [])
        w.writeheader()
        w.writerows(pair_rows)

    # Paired deltas vs baselines.
    ours = {r["item_id"]: r for r in items_by_method.get(OURS, [])}
    delta_rows = []
    for base, label in METHODS:
        if base == OURS:
            continue
        other = {r["item_id"]: r for r in items_by_method.get(base, [])}
        ids = sorted(set(ours) & set(other))
        for metric in METRICS:
            vals = []
            wins = 0
            better_sign = -1 if metric == "wer" else 1
            for item_id in ids:
                a = ours[item_id].get(metric)
                b = other[item_id].get(metric)
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                    continue
                if math.isnan(a) or math.isnan(b):
                    continue
                vals.append(a - b)
                if (a - b) * better_sign > 0:
                    wins += 1
            if vals:
                delta_rows.append({
                    "Baseline": label,
                    "Metric": metric,
                    "n": len(vals),
                    "mean_delta": f"{float(np.mean(vals)):+.4f}",
                    "wins": wins,
                })
    with (TABLES / "zh_table_deltas.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()) if delta_rows else [])
        w.writeheader()
        w.writerows(delta_rows)

    print(f"Wrote zh-expanded tables -> {TABLES}")


if __name__ == "__main__":
    main()
