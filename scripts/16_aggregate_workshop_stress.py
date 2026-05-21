#!/usr/bin/env python
"""Aggregate short/noisy prompt stress-test results."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/CYCLE_TTS")
SCORES = ROOT / "results" / "scores"
TABLES = ROOT / "results" / "tables" / "workshop_stress"
TABLES.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["short3", "noise10"]
METHODS = [
    ("b1_f5_zhx_{cond}", "F5-TTS"),
    ("b1_f5_zhx_rerank8_{cond}", "F5 + verifier/ASR rerank"),
    ("cycleadapt_zhx_final_{cond}", "CycleAdapt-Final"),
    ("cycleadapt_zhx_final_id_{cond}", "Identity-only final"),
]
FOCUS = ["cycleadapt_zhx_final_{cond}", "cycleadapt_zhx_final_id_{cond}"]
BASELINES = ["b1_f5_zhx_{cond}", "b1_f5_zhx_rerank8_{cond}"]
METRICS = ["simwavlm", "simecapa", "wer", "f0pcc", "utmos"]
LOWER_BETTER = {"wer"}


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(x)


def fmt(x) -> str:
    return "---" if not finite(x) else f"{x:.3f}"


def load_items(method: str):
    p = SCORES / f"{method}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_summary(method: str):
    p = SCORES / f"{method}.summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def mean(rows, metric):
    vals = [r.get(metric) for r in rows if finite(r.get(metric))]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    main_rows = []
    pair_rows = []
    delta_rows = []

    for cond in CONDITIONS:
        expected = sum(1 for l in (ROOT / "results" / f"eval_set_zh_workshop_{cond}.jsonl").read_text().splitlines() if l.strip())
        items_by_method = {}
        for tmpl, label in METHODS:
            key = tmpl.format(cond=cond)
            s = load_summary(key)
            if not s:
                raise SystemExit(f"Missing summary for {key}")
            if s.get("n_available") != expected:
                raise SystemExit(f"Incomplete score summary for {key}: {s.get('n_available')}/{expected}")
            rows = load_items(key)
            items_by_method[key] = rows
            for split in ["in-distrib", "zero-shot"]:
                d = s.get("by_class", {}).get(split, {})
                if d:
                    main_rows.append({
                        "Condition": cond,
                        "Method": label,
                        "Setting": split,
                        "n": d["simwavlm"]["n"],
                        "SIM-o": fmt(d["simwavlm"]["mean"]),
                        "ECAPA": fmt(d["simecapa"]["mean"]),
                        "ASR-Err": fmt(d["wer"]["mean"]),
                        "F0": fmt(d["f0pcc"]["mean"]),
                        "UTMOS": fmt(d["utmos"]["mean"]),
                    })
            for pair in sorted({r["pair_id"] for r in rows}):
                rs = [r for r in rows if r["pair_id"] == pair]
                pair_rows.append({
                    "Condition": cond,
                    "Pair": pair,
                    "Method": label,
                    "n": len(rs),
                    "SIM-o": fmt(mean(rs, "simwavlm")),
                    "ECAPA": fmt(mean(rs, "simecapa")),
                    "ASR-Err": fmt(mean(rs, "wer")),
                    "F0": fmt(mean(rs, "f0pcc")),
                    "UTMOS": fmt(mean(rs, "utmos")),
                })

        for focus_tmpl in FOCUS:
            focus = focus_tmpl.format(cond=cond)
            focus_rows = {r["item_id"]: r for r in items_by_method[focus]}
            for base_tmpl in BASELINES:
                base = base_tmpl.format(cond=cond)
                base_rows = {r["item_id"]: r for r in items_by_method[base]}
                ids = sorted(set(focus_rows) & set(base_rows))
                for metric in METRICS:
                    sign = -1.0 if metric in LOWER_BETTER else 1.0
                    vals = []
                    wins = 0
                    for item_id in ids:
                        a = focus_rows[item_id].get(metric)
                        b = base_rows[item_id].get(metric)
                        if finite(a) and finite(b):
                            benefit = sign * (a - b)
                            vals.append(benefit)
                            wins += int(benefit > 0)
                    delta_rows.append({
                        "Condition": cond,
                        "Method": dict((m.format(cond=cond), l) for m, l in METHODS)[focus],
                        "Baseline": dict((m.format(cond=cond), l) for m, l in METHODS)[base],
                        "Metric": metric,
                        "n": len(vals),
                        "mean_benefit": f"{float(np.mean(vals)):+.4f}" if vals else "nan",
                        "wins": wins,
                    })

    write_csv(TABLES / "stress_main.csv", main_rows)
    write_csv(TABLES / "stress_by_pair.csv", pair_rows)
    write_csv(TABLES / "stress_deltas.csv", delta_rows)
    print(f"Wrote workshop stress tables -> {TABLES}")


if __name__ == "__main__":
    main()
