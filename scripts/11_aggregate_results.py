#!/usr/bin/env python
"""Aggregate per-method scoring results into the main paper tables.

Reads ``results/scores/*.summary.json`` and produces:

    results/tables/exp1_main_table.csv
    results/tables/exp1_main_table.md
    results/tables/exp1_zero_shot_breakdown.csv
    results/tables/exp8_timing.csv

Statistical tests (Wilcoxon signed-rank, vs. ``ours``) are computed when
per-item JSONLs are available for both methods.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path("/home/ubuntu/CYCLE_TTS")
SCORES = ROOT / "results" / "scores"
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def load_summary(path: Path) -> Dict:
    return json.loads(path.read_text())


def load_per_item(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def fmt_ci(d: Dict) -> str:
    if d is None or d.get("mean") is None or math.isnan(d.get("mean", float("nan"))):
        return "—"
    return f"{d['mean']:.3f} [{d['ci_lo']:.3f},{d['ci_hi']:.3f}]"


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("agg")

    method_files = sorted(SCORES.glob("*.summary.json"))
    if not method_files:
        log.warning("No score summaries found in %s", SCORES)
        return

    summaries = {}
    per_item = {}
    for sp in method_files:
        method = sp.stem.replace(".summary", "")
        summaries[method] = load_summary(sp)
        per_item[method] = load_per_item(sp.parent / f"{method}.jsonl")
        log.info("Loaded method=%s  n=%d", method, summaries[method].get("n_items", 0))

    metrics = ["simwavlm", "simecapa", "f0pcc", "wer", "utmos"]

    # ---------------- Main table: per-pair_class × method × metric ----------------
    main_rows = []
    for method, summ in summaries.items():
        by_cls = summ.get("by_class", {})
        for cls, by_metric in by_cls.items():
            row = {"method": method, "pair_class": cls,
                   "n": next(iter(by_metric.values())).get("n", 0)}
            for m in metrics:
                d = by_metric.get(m, {})
                row[m] = fmt_ci(d)
            main_rows.append(row)

    csv_path = TABLES / "exp1_main_table.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["method", "pair_class", "n"] + metrics)
        w.writeheader()
        for r in main_rows:
            w.writerow(r)

    md_path = TABLES / "exp1_main_table.md"
    md = ["# Experiment 1 — Main results", "",
          "| method | class | n | SIM-o(WavLM) | SIM-o(ECAPA) | F0 PCC | WER ↓ | UTMOS ↑ |",
          "|--------|-------|---|-------------:|-------------:|------:|------:|--------:|"]
    for r in sorted(main_rows, key=lambda x: (x["pair_class"], x["method"])):
        md.append(f"| {r['method']} | {r['pair_class']} | {r['n']} | "
                  f"{r['simwavlm']} | {r['simecapa']} | {r['f0pcc']} | {r['wer']} | {r['utmos']} |")
    md_path.write_text("\n".join(md) + "\n")

    # ---------------- Zero-shot breakdown per pair ----------------
    zs_rows = []
    for method, summ in summaries.items():
        by_pair = summ.get("by_pair", {})
        for pair, by_metric in by_pair.items():
            row = {"method": method, "pair": pair,
                   "n": next(iter(by_metric.values())).get("n", 0)}
            for m in metrics:
                row[m] = fmt_ci(by_metric.get(m, {}))
            zs_rows.append(row)
    with (TABLES / "exp1_zero_shot_breakdown.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["method", "pair", "n"] + metrics)
        w.writeheader()
        for r in zs_rows:
            w.writerow(r)

    # ---------------- Timing table (Exp 8) ----------------
    timing_rows = []
    for method in summaries:
        method_dir = ROOT / "results" / "audio" / method.replace("_f5", "_f5_vanilla")
        timings_path = next(method_dir.glob("timings.jsonl"), None) if method_dir.exists() else None
        # try alt naming
        if timings_path is None:
            for alt in ["b1_f5_vanilla", "b4_xtts_v2", "b5_cosyvoice2",
                         "ours", "ablation_no_phi"]:
                p = ROOT / "results" / "audio" / alt / "timings.jsonl"
                if p.exists() and (alt in method or method in alt):
                    timings_path = p; break
        if timings_path is None or not timings_path.exists():
            continue
        rtfs = []
        elapsed = []
        for line in timings_path.read_text().split("\n"):
            if not line.strip(): continue
            d = json.loads(line)
            if isinstance(d.get("rtf"), (int, float)) and d["rtf"] == d["rtf"]:
                rtfs.append(d["rtf"])
            if isinstance(d.get("elapsed_sec"), (int, float)):
                elapsed.append(d["elapsed_sec"])
        if rtfs:
            arr = np.asarray(rtfs)
            timing_rows.append({
                "method": method,
                "n": len(arr),
                "mean_rtf": float(arr.mean()),
                "median_rtf": float(np.median(arr)),
                "p95_rtf": float(np.quantile(arr, 0.95)),
                "mean_elapsed_sec": float(np.mean(elapsed)) if elapsed else float("nan"),
            })
    with (TABLES / "exp8_timing.csv").open("w") as f:
        if timing_rows:
            w = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
            w.writeheader()
            for r in timing_rows:
                w.writerow(r)

    # ---------------- Significance tests (Wilcoxon vs. ours) ----------------
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except ImportError:
        wilcoxon = None
    sig_lines = ["# Significance tests (Wilcoxon signed-rank, paired by item_id, vs. `ours`)",
                  "", "| method | metric | n | W | p | direction |", "|--------|--------|---|--:|---:|-----------|"]
    if "ours" in per_item and wilcoxon is not None:
        ours_idx = {r["item_id"]: r for r in per_item["ours"]}
        for method, rows in per_item.items():
            if method == "ours": continue
            for m in metrics:
                paired_ours, paired_other = [], []
                for r in rows:
                    o = ours_idx.get(r["item_id"])
                    if o is None: continue
                    if not isinstance(r.get(m), (int, float)) or not isinstance(o.get(m), (int, float)):
                        continue
                    if math.isnan(r[m]) or math.isnan(o[m]):
                        continue
                    paired_ours.append(o[m]); paired_other.append(r[m])
                if len(paired_ours) < 5:
                    sig_lines.append(f"| {method} | {m} | {len(paired_ours)} | — | — | — |")
                    continue
                a = np.asarray(paired_ours); b = np.asarray(paired_other)
                try:
                    res = wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
                    direction = "ours better" if (a.mean() > b.mean() and m != "wer") or (a.mean() < b.mean() and m == "wer") else f"{method} better"
                    sig_lines.append(f"| {method} | {m} | {len(a)} | {res.statistic:.1f} | {res.pvalue:.2e} | {direction} |")
                except Exception as e:
                    sig_lines.append(f"| {method} | {m} | {len(a)} | err: {e} | — | — |")
    (TABLES / "significance.md").write_text("\n".join(sig_lines) + "\n")

    log.info("Wrote tables -> %s", TABLES)


if __name__ == "__main__":
    main()
