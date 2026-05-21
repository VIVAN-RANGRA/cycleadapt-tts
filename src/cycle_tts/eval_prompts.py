"""Deterministic held-out evaluation prompt set.

Every baseline + our method evaluates on **the exact same prompts and target
texts**.  The prompt set is reproducible from the manifests + a fixed seed.

Prompt sourcing
---------------
* ``L1 = en`` → VCTK held-out speakers (110 unseen by training).
* ``L1 = zh`` → FLEURS-zh utterances (single-utterance "speakers" by FLEURS
  convention; treat utterance id as speaker id since the entire dataset is
  held out from training).

Target text sourcing
--------------------
For each L2 language we draw target texts from the corresponding FLEURS test
TSV (or VCTK normalized text for L2 = en).  This guarantees:

* texts are natural, well-formed and length-controlled,
* there is NO speaker overlap between prompt and target text (target text
  comes from a different utterance / corpus),
* texts are reusable across baselines so the comparison is fair.

Per (L1, L2) pair we sample ``n_speakers`` distinct prompts; for each prompt
we sample a single target text.  Sampling is seeded by hashing
``(seed, L1, L2, "slot k")``.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .data import DATA, ManifestIndex


@dataclass(frozen=True)
class EvalItem:
    """A single (prompt, gen_text) evaluation trial."""
    pair_id: str                # "{L1}_{L2}"
    pair_class: str             # "in-distrib" | "zero-shot"
    L1: str
    L2: str
    slot: int                   # 0..n_speakers-1
    speaker_id: str             # opaque id of the prompt speaker
    prompt_wav: str             # absolute path to prompt audio
    prompt_text: str            # transcript of prompt (L1)
    prompt_sr: int
    gen_text: str               # text to generate (L2)
    target_id: str              # opaque id of the generation text (for WER)


def _seeded_rng(*parts) -> random.Random:
    h = hashlib.sha256("__".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _load_manifest_rows(jsonl: Path) -> List[Dict]:
    rows = []
    if not jsonl.exists():
        return rows
    for line in jsonl.read_text().split("\n"):
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _vctk_speakers_index(manifest_rows: List[Dict]) -> Dict[str, List[Dict]]:
    """VCTK rows grouped by speaker."""
    out: Dict[str, List[Dict]] = {}
    for r in manifest_rows:
        out.setdefault(r["speaker_id"], []).append(r)
    return out


def _fleurs_rows_by_lang(rows: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(r["lang"], []).append(r)
    return out


def build_eval_set(
    eval_pairs: List[Tuple[str, str, str]],
    *,
    n_speakers: int = 25,
    seed: int = 1337,
    vctk_eval_jsonl: Optional[Path] = None,
    fleurs_eval_jsonl: Optional[Path] = None,
    target_max_utf8_bytes: Optional[int] = None,
) -> List[EvalItem]:
    """Build the deterministic eval set.

    Parameters
    ----------
    eval_pairs
        Iterable of ``(L1, L2, pair_class)`` tuples (from ``cfg.eval_lang_pairs``).
    n_speakers
        Distinct speakers per (L1, L2) pair.
    seed
        Master sampling seed.
    target_max_utf8_bytes
        Optional upper bound on target-text byte length. This keeps F5's
        duration heuristic below the DiT positional limit for languages whose
        scripts use multiple UTF-8 bytes per character (notably Hindi).
    """
    vctk_eval_jsonl = vctk_eval_jsonl or (DATA / "manifests" / "vctk_eval.jsonl")
    fleurs_eval_jsonl = fleurs_eval_jsonl or (DATA / "manifests" / "fleurs_eval.jsonl")

    vctk_rows = _load_manifest_rows(vctk_eval_jsonl)
    fleurs_rows = _load_manifest_rows(fleurs_eval_jsonl)
    fleurs_by_lang = _fleurs_rows_by_lang(fleurs_rows)
    vctk_by_speaker = _vctk_speakers_index(vctk_rows)
    vctk_speakers_sorted = sorted(vctk_by_speaker.keys())  # deterministic order

    items: List[EvalItem] = []
    for L1, L2, pclass in eval_pairs:
        # ---- pick PROMPT source ----
        if L1 == "en":
            rng = _seeded_rng(seed, "prompt_speakers", L1)
            spk_choices = list(vctk_speakers_sorted)
            rng.shuffle(spk_choices)
            if len(spk_choices) < n_speakers:
                raise ValueError(f"Not enough VCTK speakers ({len(spk_choices)}) for {L1}")
            spk_choices = spk_choices[:n_speakers]
            prompt_rows = []
            for spk in spk_choices:
                # Each VCTK row IS one .flac utterance — pick one per speaker.
                spk_rows = [r for r in vctk_by_speaker[spk] if r["shard"].endswith(".flac")]
                row_rng = _seeded_rng(seed, "prompt_utt", L1, spk)
                prompt_rows.append(row_rng.choice(spk_rows))
        elif L1 == "zh":
            rng = _seeded_rng(seed, "prompt_speakers", L1)
            cands = fleurs_by_lang.get("zh", [])
            if len(cands) < n_speakers:
                raise ValueError(f"Not enough FLEURS-zh rows ({len(cands)}) for {L1}")
            prompt_rows = rng.sample(cands, n_speakers)
        else:
            raise ValueError(f"Unsupported L1 in eval: {L1}")

        # ---- pick TARGET text per slot ----
        text_rng = _seeded_rng(seed, "target_text", L1, L2)
        target_pool = []
        if L2 == "en":
            # Use FLEURS-en sentences as English target text (held out, public,
            # multilingual-comparable for WER under whisper).
            target_pool = fleurs_by_lang.get("en", [])
        else:
            target_pool = fleurs_by_lang.get(L2, [])
        if target_max_utf8_bytes is not None:
            target_pool = [
                r for r in target_pool
                if len(str(r.get("text", "")).encode("utf-8")) <= target_max_utf8_bytes
            ]
        if len(target_pool) < n_speakers:
            raise ValueError(f"Not enough target texts for {L2} ({len(target_pool)})")

        targets = text_rng.sample(target_pool, n_speakers)

        # ---- emit items ----
        for slot, (prow, trow) in enumerate(zip(prompt_rows, targets)):
            items.append(EvalItem(
                pair_id=f"{L1}_{L2}",
                pair_class=pclass,
                L1=L1, L2=L2, slot=slot,
                speaker_id=prow["speaker_id"],
                prompt_wav=prow["shard"],
                prompt_text=prow["text"],
                prompt_sr=int(prow.get("sample_rate") or 24_000),
                gen_text=trow["text"],
                target_id=trow.get("speaker_id") or trow.get("row_or_path", ""),
            ))

    return items


def save_eval_set(items: List[EvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it.__dict__) + "\n")


def load_eval_set(path: Path) -> List[EvalItem]:
    out = []
    for line in Path(path).read_text().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(EvalItem(**d))
    return out
