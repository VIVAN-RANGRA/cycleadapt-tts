#!/usr/bin/env python
"""Ablation A1 — no learned loss weighting φ.

Trains the full CycleAdapt-TTS stack but with φ FROZEN at its initial
(uniform) state.  All other knobs match the main run.

This validates the contribution of the learned loss weighter to the final
quality.  Runs ``M_ablation`` meta-iters (typically ~60-70% of main) to fit
the budget, then writes a checkpoint to
``checkpoints/ablation_no_phi/final.pt``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

ROOT = Path("/home/ubuntu/CYCLE_TTS")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party_f5" / "src"))

from cycle_tts.config import CycleAdaptConfig  # noqa: E402
from cycle_tts.data import ManifestIndex  # noqa: E402
from cycle_tts.f5_wrapper import F5CycleWrapper, load_f5tts_model  # noqa: E402
from cycle_tts.feature_extractors import F0Extractor, MelSpecExtractor, WavLMSpeakerEncoder  # noqa: E402
from cycle_tts.iaa import IdentityAlignmentAdapter  # noqa: E402
from cycle_tts.loss_weighter import LossWeighter  # noqa: E402
from cycle_tts.meta_optimizer import LearnedOptimizer  # noqa: E402
from cycle_tts.meta_trainer import MetaTrainer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=1200)
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--run-name", type=str, default="ablation_no_phi")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log = logging.getLogger("ablation_no_phi")

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

    cfg = CycleAdaptConfig(run_name=args.run_name)
    cfg.train.M = args.M
    cfg.train.B = args.B
    cfg.cycle.K_train = args.K

    log.info("ABLATION — no φ.  M=%d B=%d K=%d  run_name=%s", cfg.train.M, cfg.train.B, cfg.cycle.K_train, cfg.run_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading F5-TTS …")
    cfm, vocoder, vocab_char_map = load_f5tts_model(device=device, bf16=True)
    f5 = F5CycleWrapper(cfm, vocoder, vocab_char_map)

    iaa = IdentityAlignmentAdapter(cfg.iaa, cfm.transformer)
    iaa.freeze_base()
    psi = LearnedOptimizer(cfg.psi)
    phi = LossWeighter(cfg.phi)

    # CRITICAL: freeze φ.  No grad will reach it; the inner loop still calls
    # ``phi(losses, l1, l2)`` and uses the (uniform) softmax weights.
    for p_ in phi.parameters():
        p_.requires_grad = False
    phi.eval()

    spk_encoder = WavLMSpeakerEncoder(device=device)
    f0_extractor = F0Extractor(device=device)
    mel_extractor = MelSpecExtractor(cfm.mel_spec)
    manifest = ManifestIndex.from_jsonl_files(sorted(Path(cfg.data.manifest_dir).glob("*train.jsonl")))

    trainer = MetaTrainer(
        cfg=cfg, iaa=iaa, psi=psi, phi=phi, f5=f5,
        spk_encoder=spk_encoder, f0_extractor=f0_extractor,
        mel_extractor=mel_extractor, manifest=manifest,
        device=device, log_wandb=False, log_tb=True,
    )

    # Disable the φ optimizer so even if a stray gradient slipped through it
    # would have no effect.
    trainer.phi_opt = torch.optim.SGD(phi.parameters(), lr=0.0)

    trainer.fit()


if __name__ == "__main__":
    main()
