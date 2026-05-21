#!/usr/bin/env python
"""Diagnose LoRA collapse: compare checkpoint θ₀ vs pristine init on one utterance."""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path("/home/ubuntu/CYCLE_TTS")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party_f5" / "src"))

from cycle_tts.config import CycleAdaptConfig
from cycle_tts.eval_prompts import load_eval_set
from cycle_tts.f5_wrapper import F5CycleWrapper, load_f5tts_model
from cycle_tts.iaa import IdentityAlignmentAdapter
from cycle_tts.lora_stability import clamp_init_params
from cycle_tts.meta_trainer import _lora_param_name_to_path, _make_override


def stats(label: str, A_list, B_list, scaling: float) -> None:
    A = torch.cat([t.float().flatten() for t in A_list])
    B = torch.cat([t.float().flatten() for t in B_list])
    print(f"{label:20s}  A: std={A.std():.4f} max={A.abs().max():.3f}  "
          f"B: std={B.std():.4f} max={B.abs().max():.3f}  "
          f"eff_scale~{scaling * A.norm() * B.norm() / max(len(A_list), 1):.4f}")


def gen_peak(f5, iaa, paths, A, B, item, nfe=16) -> float:
    import torchaudio
    wav, sr = torchaudio.load(item.prompt_wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    wav = wav.squeeze(0)
    override = _make_override(iaa, A, B, paths)
    with torch.inference_mode():
        out = f5.generate_eval(wav, item.prompt_text, item.gen_text, nfe_step=nfe, seed=42)
    w = out.wave.squeeze().float().cpu().numpy()
    return float(abs(w).max()), w


def main() -> None:
    cfg = CycleAdaptConfig()
    device = "cuda"
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else cfg.ckpt_dir / "final.pt"

    cfm, voc, vocab = load_f5tts_model(device=device, bf16=True)
    f5 = F5CycleWrapper(cfm, voc, vocab)
    iaa = IdentityAlignmentAdapter(cfg.iaa, cfm.transformer)
    iaa.freeze_base()
    paths = _lora_param_name_to_path(iaa)
    n_rec = len(iaa.records)
    scaling = cfg.iaa.alpha / cfg.iaa.rank

    item = load_eval_set(ROOT / "results" / "eval_set.jsonl")[0]

    # Pristine θ₀
    A_pri = [iaa.init_params[i].detach().clone() for i in range(n_rec)]
    B_pri = [iaa.init_params[n_rec + i].detach().clone() for i in range(n_rec)]
    stats("pristine", A_pri, B_pri, scaling)
    peak_pri, _ = gen_peak(f5, iaa, paths, A_pri, B_pri, item)
    print(f"  -> peak amplitude (pristine): {peak_pri:.4f}")

    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        iaa.load_state_dict_iaa_only(payload["iaa"])
        A_ck = [iaa.init_params[i].detach().clone() for i in range(n_rec)]
        B_ck = [iaa.init_params[n_rec + i].detach().clone() for i in range(n_rec)]
        stats(f"ckpt {ckpt_path.name}", A_ck, B_ck, scaling)
        peak_ck, _ = gen_peak(f5, iaa, paths, A_ck, B_ck, item)
        print(f"  -> peak amplitude (checkpoint): {peak_ck:.4f}")

        clamp_init_params(iaa.init_params, n_rec, cfg.stab, cfg.iaa)
        A_cl = [iaa.init_params[i].detach().clone() for i in range(n_rec)]
        B_cl = [iaa.init_params[n_rec + i].detach().clone() for i in range(n_rec)]
        stats("after clamp", A_cl, B_cl, scaling)
        peak_cl, w_cl = gen_peak(f5, iaa, paths, A_cl, B_cl, item)
        print(f"  -> peak amplitude (clamped): {peak_cl:.4f}")
        sf.write(ROOT / "results" / "trace_clamped_sample.wav", w_cl, 24000)
        print(f"  wrote {ROOT / 'results' / 'trace_clamped_sample.wav'}")


if __name__ == "__main__":
    main()
