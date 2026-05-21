#!/usr/bin/env python
"""EMNLP Findings tables: main results, ablations, significance vs ours_emnlp."""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path("/home/ubuntu/CYCLE_TTS")
SCORES = ROOT / "results" / "scores"
TABLES = ROOT / "results" / "tables" / "emnlp"
TABLES.mkdir(parents=True, exist_ok=True)

# Paper-facing method order and display names.
METHOD_ORDER = [
    ("b1_f5", "F5-TTS (baseline)"),
    ("b1_f5_rerank8", "F5-TTS + verifier rerank"),
    ("b2_random_adam", "F5 + random LoRA + Adam TTT"),
    ("b3_emnlp", "Meta-init $\\theta_0$ only"),
    ("ours_emnlp", "CycleAdapt w/ learned $\\phi$"),
    ("cycleadapt_final", "\\textbf{CycleAdapt-Final}"),
    ("cycleadapt_final_id", "Identity-only final"),
    ("a1_no_phi", "w/o $\\phi$"),
    ("a3_no_cycle", "w/o cycle"),
    ("id_only_ttt", "TTT w/ $\\mathcal{L}_{id}$ only"),
]

OURS_KEY = "cycleadapt_final" if (SCORES / "cycleadapt_final.summary.json").exists() else "ours_emnlp"
BASELINE_KEY = "b1_f5"
METRICS = ["simwavlm", "simecapa", "f0pcc", "wer", "utmos"]


def load_summary(method: str) -> Optional[Dict]:
    p = SCORES / f"{method}.summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_items(method: str) -> List[Dict]:
    p = SCORES / f"{method}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def fmt(d: Dict, nd: int = 3) -> str:
    if not d or d.get("mean") is None or (isinstance(d["mean"], float) and math.isnan(d["mean"])):
        return "---"
    return f"{d['mean']:.{nd}f}"


def wilcoxon_vs_ours(ours_items: Dict[str, Dict], other_items: Dict[str, Dict], metric: str) -> str:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return "---"
    a, b = [], []
    for k, o in ours_items.items():
        t = other_items.get(k)
        if t is None:
            continue
        if metric not in o or metric not in o:
            continue
        ov, tv = o[metric], t[metric]
        if not isinstance(ov, (int, float)) or not isinstance(tv, (int, float)):
            continue
        if math.isnan(ov) or math.isnan(tv):
            continue
        a.append(ov)
        b.append(tv)
    if len(a) < 8:
        return "---"
    stat, p = wilcoxon(a, b, alternative="two-sided")
    better = "ours" if (np.mean(a) > np.mean(b) and metric != "wer") or (np.mean(a) < np.mean(b) and metric == "wer") else "other"
    sig = "*" if p < 0.05 else ""
    return f"p={p:.3f}{sig} ({better})"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summaries = {m: load_summary(m) for m, _ in METHOD_ORDER}
    summaries.update({p.stem.replace(".summary", ""): load_summary(p.stem.replace(".summary", ""))
                      for p in SCORES.glob("*.summary.json")})

    ours_items = {r["item_id"]: r for r in load_items(OURS_KEY)} if load_items(OURS_KEY) else {}

    # ---- Table 1: Main (zero-shot + in-distrib) ----
    rows = []
    for key, label in METHOD_ORDER:
        s = summaries.get(key)
        if s is None:
            continue
        for cls in ["in-distrib", "zero-shot"]:
            bm = s.get("by_class", {}).get(cls, {})
            rows.append({
                "Method": label,
                "Setting": cls,
                "SIM-o": fmt(bm.get("simwavlm")),
                "ECAPA": fmt(bm.get("simecapa")),
                "ASR-Err": fmt(bm.get("wer")),
                "F0": fmt(bm.get("f0pcc")),
                "UTMOS": fmt(bm.get("utmos")),
            })

    with (TABLES / "table1_main.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    # LaTeX snippet
    lines = [
        "% Auto-generated EMNLP Table 1",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Method & Setting & SIM-o & ECAPA & ASR-Err & F0 \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(f"{r['Method']} & {r['Setting']} & {r['SIM-o']} & {r['ECAPA']} & {r['ASR-Err']} & {r['F0']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "table1_main.tex").write_text("\n".join(lines) + "\n")

    # ---- Table 2: Per language pair (zero-shot) for ours vs b1 ----
    pair_rows = []
    for key in [BASELINE_KEY, OURS_KEY]:
        items = load_items(key)
        by_pair = defaultdict(list)
        for r in items:
            if r.get("pair_class") == "zero-shot":
                by_pair[r["pair_id"]].append(r.get("simwavlm", float("nan")))
        for pair, vals in sorted(by_pair.items()):
            v = [x for x in vals if isinstance(x, (int, float)) and x == x]
            pair_rows.append({
                "method": key,
                "pair": pair,
                "simwavlm_mean": float(np.mean(v)) if v else float("nan"),
                "n": len(v),
            })
    with (TABLES / "table2_per_pair_zs.csv").open("w", newline="") as f:
        if pair_rows:
            w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)

    # ---- Significance vs ours ----
    sig_lines = [f"# Wilcoxon vs {OURS_KEY} (paired by item_id)", ""]
    sig_lines.append("| baseline | metric | test |")
    sig_lines.append("|----------|--------|------|")
    for key, label in METHOD_ORDER:
        if key == OURS_KEY:
            continue
        other = {r["item_id"]: r for r in load_items(key)}
        if not other:
            continue
        for m in ["simwavlm", "wer"]:
            sig_lines.append(f"| {key} | {m} | {wilcoxon_vs_ours(ours_items, other, m)} |")
    (TABLES / "table_significance.md").write_text("\n".join(sig_lines) + "\n")

    logging.info("Wrote EMNLP tables -> %s", TABLES)


if __name__ == "__main__":
    main()
