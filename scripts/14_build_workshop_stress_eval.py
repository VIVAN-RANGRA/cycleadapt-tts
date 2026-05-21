#!/usr/bin/env python
"""Build short/noisy prompt stress eval sets from zh-expanded eval."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torchaudio

ROOT = Path("/home/ubuntu/CYCLE_TTS")
sys.path.insert(0, str(ROOT / "src"))

from cycle_tts.eval_prompts import EvalItem, load_eval_set, save_eval_set  # noqa: E402


def load_24k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 24_000:
        wav = torchaudio.functional.resample(wav, sr, 24_000)
    return wav.squeeze(0).float()


def save_24k(wav: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().float().clamp(-0.99, 0.99).unsqueeze(0)
    torchaudio.save(str(path), wav, 24_000)


def truncate_text(text: str, ratio: float) -> str:
    text = text.strip()
    if ratio >= 0.98:
        return text
    n = max(4, int(math.ceil(len(text) * max(0.05, ratio))))
    return text[: min(n, len(text))]


def make_short(item: EvalItem, out_wav: Path, seconds: float) -> EvalItem:
    wav = load_24k(item.prompt_wav)
    orig_sec = wav.numel() / 24_000
    keep = min(wav.numel(), int(seconds * 24_000))
    short = wav[:keep]
    save_24k(short, out_wav)
    ratio = min(1.0, max(1e-6, short.numel() / max(wav.numel(), 1)))
    return replace(
        item,
        speaker_id=f"{item.speaker_id}__short{seconds:g}",
        prompt_wav=str(out_wav),
        prompt_text=truncate_text(item.prompt_text, ratio),
        prompt_sr=24_000,
    )


def make_noisy(item: EvalItem, out_wav: Path, snr_db: float, seed: int) -> EvalItem:
    wav = load_24k(item.prompt_wav)
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(wav.shape, generator=gen)
    sig_power = wav.pow(2).mean().clamp_min(1e-8)
    noise_power = noise.pow(2).mean().clamp_min(1e-8)
    scale = torch.sqrt(sig_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    noisy = (wav + scale * noise).clamp(-0.99, 0.99)
    save_24k(noisy, out_wav)
    return replace(
        item,
        speaker_id=f"{item.speaker_id}__noise{snr_db:g}",
        prompt_wav=str(out_wav),
        prompt_sr=24_000,
    )


def select_subset(items: list[EvalItem], n_per_pair: int) -> list[EvalItem]:
    by_pair: dict[str, list[EvalItem]] = {}
    for it in items:
        by_pair.setdefault(it.pair_id, []).append(it)
    out = []
    for pair in sorted(by_pair):
        out.extend(sorted(by_pair[pair], key=lambda x: x.slot)[:n_per_pair])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(ROOT / "results" / "eval_set_zh_expanded.jsonl"))
    p.add_argument("--n-per-pair", type=int, default=10)
    p.add_argument("--short-sec", type=float, default=3.0)
    p.add_argument("--noise-snr-db", type=float, default=10.0)
    p.add_argument("--variants", nargs="+", default=["short3", "noise10"],
                   choices=["short3", "noise10"])
    args = p.parse_args()

    src = load_eval_set(Path(args.source))
    subset = select_subset(src, args.n_per_pair)
    out_root = ROOT / "results" / "prompts" / "workshop_stress"

    for variant in args.variants:
        rows = []
        for it in subset:
            item_id = f"{it.pair_id}_{it.slot:03d}"
            out_wav = out_root / variant / f"{item_id}.wav"
            if variant == "short3":
                rows.append(make_short(it, out_wav, args.short_sec))
            elif variant == "noise10":
                rows.append(make_noisy(it, out_wav, args.noise_snr_db, seed=1337 + it.slot + hash(it.pair_id) % 10000))
            else:
                raise ValueError(variant)
        out_path = ROOT / "results" / f"eval_set_zh_workshop_{variant}.jsonl"
        save_eval_set(rows, out_path)
        print(f"Wrote {len(rows)} rows -> {out_path}")
        print(json.dumps({"variant": variant, "pairs": sorted({r.pair_id for r in rows})}))


if __name__ == "__main__":
    main()
