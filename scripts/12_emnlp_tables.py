#!/usr/bin/env python
"""Build EMNLP Findings-ready tables from all scored methods."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path("/home/ubuntu/CYCLE_TTS")
TABLES = ROOT / "results" / "tables"
SCORES = ROOT / "results" / "scores"

# Display order for the paper (main + ablations).
MAIN_METHODS = [
    ("Vanilla F5-TTS (B1)", "b1_f5"),
    ("+ random LoRA + Adam TTT (B2)", "b2_random_adam"),
    ("+ meta-init $\\theta_0$ only (B3)", "b3_v2_nfe32"),
    ("CycleAdapt (ours, v2)", "ours_v2_nfe32"),
    ("CycleAdapt + $\\mathcal{L}_{id}$-only TTT", "ours_v2_lid_nfe32"),
    ("CycleAdapt v3", "ours_v3_nfe32"),
    ("CycleAdapt v3 + $\\mathcal{L}_{id}$-only TTT", "ours_v3_lid_nfe32"),
]

ABLATION_METHODS = [
    ("Full CycleAdapt v2 (NFE=16)", "ours_v2"),
    ("Fair NFE=32", "ours_v2_nfe32"),
    ("$\\mathcal{L}_{id}$-only TTT", "ours_v2_lid_nfe32"),
    ("No cycle (train ablation)", "a3_no_cycle"),
    ("No $\\phi$ (test ablation)", "a1_no_phi_test"),
]


def load_summary(method: str) -> Optional[Dict]:
    p = SCORES / f"{method}.summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt_metric(d: Dict, key: str, higher_better: bool = True) -> str:
    x = d.get(key, {})
    m, lo, hi = x.get("mean"), x.get("ci_lo"), x.get("ci_hi")
    if m is None or m != m:
        return "—"
    arrow = "" if higher_better else ""
    return f"{m:.3f}"


def build_main_table() -> None:
    rows = []
    for label, mid in MAIN_METHODS:
        s = load_summary(mid)
        if s is None:
            continue
        for cls in ["in-distrib", "zero-shot"]:
            d = s.get("by_class", {}).get(cls, {})
            rows.append({
                "method": label,
                "class": cls,
                "n": d.get("simwavlm", {}).get("n", ""),
                "sim_wavlm": fmt_metric(d, "simwavlm"),
                "sim_ecapa": fmt_metric(d, "simecapa"),
                "wer": fmt_metric(d, "wer", higher_better=False),
                "f0_pcc": fmt_metric(d, "f0pcc"),
                "utmos": fmt_metric(d, "utmos"),
            })

    csv_path = TABLES / "emnlp_main.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    md = ["# Table 1 — Cross-lingual voice cloning (EMNLP Findings)",
          "",
          "| Method | Setting | n | SIM-o (WavLM) | SIM-o (ECAPA) | WER ↓ | F0 PCC | MOS |",
          "|--------|---------|--:|--------------:|--------------:|------:|------:|----:|"]
    for r in rows:
        md.append(
            f"| {r['method']} | {r['class']} | {r['n']} | {r['sim_wavlm']} | {r['sim_ecapa']} | "
            f"{r['wer']} | {r['f0_pcc']} | {r['utmos']} |"
        )
    (TABLES / "emnlp_main.md").write_text("\n".join(md) + "\n")

    # LaTeX snippet
    tex = [
        "% Auto-generated — paste into EMNLP paper",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Cross-lingual speaker similarity and intelligibility. "
        "Trained on EN+ZH only; zero-shot rows are ES/FR/DE/HI/JA.}",
        "\\label{tab:main}",
        "\\small",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Method & Split & SIM-o$\\uparrow$ & ECAPA$\\uparrow$ & WER$\\downarrow$ & F0$\\uparrow$ \\\\",
        "\\midrule",
    ]
    for label, mid in MAIN_METHODS:
        s = load_summary(mid)
        if not s:
            continue
        zs = s["by_class"].get("zero-shot", {})
        tex.append(
            f"{label} & ZS & "
            f"{zs.get('simwavlm',{}).get('mean',0):.3f} & "
            f"{zs.get('simecapa',{}).get('mean',0):.3f} & "
            f"{zs.get('wer',{}).get('mean',0):.3f} & "
            f"{zs.get('f0pcc',{}).get('mean',0):.3f} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (TABLES / "emnlp_main.tex").write_text("\n".join(tex) + "\n")


def build_per_language() -> None:
    """Zero-shot breakdown by target language."""
    methods = ["b1_f5", "ours_v2_nfe32", "ours_v2_lid_nfe32", "ours_v3_nfe32"]
    pairs = ["en_es", "en_fr", "en_de", "en_hi", "en_ja", "zh_ja"]
    rows = []
    for pair in pairs:
        row = {"pair": pair}
        for m in methods:
            s = load_summary(m)
            if not s:
                row[m] = ""
                continue
            d = s.get("by_pair", {}).get(pair, {}).get("simwavlm", {})
            row[m] = f"{d.get('mean', float('nan')):.3f}" if d else ""
        rows.append(row)
    with (TABLES / "emnlp_per_language.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair"] + methods)
        w.writeheader()
        w.writerows(rows)


def build_comparison_vs_b1() -> None:
    """Paired delta ours vs B1 on zero-shot SIM-o."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    b1_items = {}
    for line in (SCORES / "b1_f5.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("pair_class") == "zero-shot":
                b1_items[r["item_id"]] = r["simwavlm"]
    lines = ["# Paired comparison vs B1 (zero-shot SIM-o WavLM)", "",
             "| Method | mean Δ vs B1 | wins/250 | Wilcoxon p |",
             "|--------|------------:|---------:|-----------:|"]
    for label, mid in MAIN_METHODS[3:]:
        p = SCORES / f"{mid}.jsonl"
        if not p.exists():
            continue
        paired_b, paired_o = [], []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["pair_class"] != "zero-shot":
                continue
            iid = r["item_id"]
            if iid in b1_items:
                paired_b.append(b1_items[iid])
                paired_o.append(r["simwavlm"])
        if len(paired_b) < 10:
            continue
        import numpy as np
        b, o = np.array(paired_b), np.array(paired_o)
        delta = o - b
        wins = int((delta > 0).sum())
        pval = "—"
        if wilcoxon and len(delta) >= 5:
            try:
                res = wilcoxon(o, b, alternative="two-sided")
                pval = f"{res.pvalue:.2e}"
            except Exception:
                pass
        lines.append(f"| {label} | {delta.mean():+.3f} | {wins}/{len(delta)} | {pval} |")
    (TABLES / "emnlp_vs_b1.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    build_main_table()
    build_per_language()
    build_comparison_vs_b1()
    print(f"Wrote EMNLP tables -> {TABLES}")


if __name__ == "__main__":
    main()
