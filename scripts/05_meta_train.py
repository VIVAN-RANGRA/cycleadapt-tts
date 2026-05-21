#!/usr/bin/env python
"""Launch CycleAdapt-TTS meta-training.

Usage examples
--------------

# Tiny prototype run for sanity-checking (≈ 25 minutes).
python scripts/05_meta_train.py --M 30 --B 2 --run-name proto_M30

# Full 2-day-A100 production run (≈ 13 hours pure training).
python scripts/05_meta_train.py --M 900 --B 4 --run-name cycleadapt_v1 --wandb

Hyperparameters not exposed by the CLI come from ``cycle_tts.config.CycleAdaptConfig``.
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

from cycle_tts.config import CycleAdaptConfig
from cycle_tts.data import ManifestIndex
from cycle_tts.episode_sampler import EpisodeSampler  # noqa: F401  (imported via trainer)
from cycle_tts.f5_wrapper import F5CycleWrapper, load_f5tts_model
from cycle_tts.feature_extractors import F0Extractor, MelSpecExtractor, WavLMSpeakerEncoder
from cycle_tts.iaa import IdentityAlignmentAdapter
from cycle_tts.loss_weighter import LossWeighter
from cycle_tts.meta_optimizer import LearnedOptimizer
from cycle_tts.meta_trainer import MetaTrainer
from cycle_tts.speed import apply_global_speed_flags, compile_dit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=None, help="meta-iterations (overrides cfg.train.M)")
    p.add_argument("--B", type=int, default=None, help="episodes per meta-batch")
    p.add_argument("--K", type=int, default=None, help="inner-loop steps")
    p.add_argument("--nfe-fwd", type=int, default=None)
    p.add_argument("--nfe-cyc", type=int, default=None)
    p.add_argument("--nfe-outer", type=int, default=None)
    p.add_argument("--lr-optim", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", type=str, default="cycleadapt_v1")
    p.add_argument("--resume", type=str, default="", help="path to checkpoint to resume from")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-tb", action="store_true")
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(DiT) — inference/eval only; breaks FOMAML training (retain_graph)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("meta_train")

    # Safe global speed-ups.  Avoid cudnn.benchmark — our inputs are
    # variable-shape so autotuning each new shape costs more than the
    # gain.  TF32 + fused SDPA are pure wins.
    apply_global_speed_flags()
    log.info("Speed-up flags: TF32 on, SDPA fused kernels on")

    cfg = CycleAdaptConfig(run_name=args.run_name)
    if args.M is not None: cfg.train.M = args.M
    if args.B is not None: cfg.train.B = args.B
    if args.K is not None: cfg.cycle.K_train = args.K
    if args.nfe_fwd is not None: cfg.cycle.nfe_forward_inner = args.nfe_fwd
    if args.nfe_cyc is not None: cfg.cycle.nfe_cycle_inner = args.nfe_cyc
    if args.nfe_outer is not None: cfg.cycle.nfe_outer = args.nfe_outer
    if args.lr_optim is not None: cfg.train.lr_optim = args.lr_optim
    if args.seed is not None: cfg.train.seed = args.seed

    log.info("Configuration:")
    log.info("  M=%d B=%d K=%d  NFE(fwd/cyc/outer)=%d/%d/%d  lr=%g  seed=%d",
             cfg.train.M, cfg.train.B, cfg.cycle.K_train,
             cfg.cycle.nfe_forward_inner, cfg.cycle.nfe_cycle_inner, cfg.cycle.nfe_outer,
             cfg.train.lr_optim, cfg.train.seed)
    log.info("  ckpt_dir=%s log_dir=%s", cfg.ckpt_dir, cfg.log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Build models
    # ------------------------------------------------------------------
    log.info("Loading F5-TTS backbone …")
    cfm, vocoder, vocab_char_map = load_f5tts_model(device=device, bf16=True)
    f5 = F5CycleWrapper(cfm, vocoder, vocab_char_map)
    if args.compile:
        log.warning(
            "--compile is not compatible with meta-training (FOMAML retain_graph). "
            "Use --compile in scripts/08_method_ours_ttt.py for eval instead."
        )

    log.info("Building IAA (LoRA) …")
    iaa = IdentityAlignmentAdapter(cfg.iaa, cfm.transformer)
    iaa.freeze_base()
    log.info("  IAA records=%d  adapter params=%d  init params (fp32)=%d",
             len(iaa.records), iaa.n_adapter_params(),
             sum(p.numel() for p in iaa.init_parameters()))

    log.info("Building ψ (learned optimizer) …")
    psi = LearnedOptimizer(cfg.psi)
    log.info("  ψ params=%d", psi.num_params())

    log.info("Building φ (loss weighter) …")
    phi = LossWeighter(cfg.phi)
    log.info("  φ params=%d", sum(p.numel() for p in phi.parameters()))

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------
    log.info("Loading feature extractors (WavLM, MelSpec, F0) …")
    spk_encoder = WavLMSpeakerEncoder(device=device)
    f0_extractor = F0Extractor(device=device)
    mel_extractor = MelSpecExtractor(cfm.mel_spec)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    log.info("Loading manifests …")
    manifest_paths = sorted(Path(cfg.data.manifest_dir).glob("*train.jsonl"))
    log.info("  manifest files: %s", [p.name for p in manifest_paths])
    manifest = ManifestIndex.from_jsonl_files(manifest_paths)
    log.info("  total rows=%d  EN speakers=%d  ZH speakers=%d",
             len(manifest.rows), manifest.n_speakers("en"), manifest.n_speakers("zh"))

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = MetaTrainer(
        cfg=cfg,
        iaa=iaa,
        psi=psi,
        phi=phi,
        f5=f5,
        spk_encoder=spk_encoder,
        f0_extractor=f0_extractor,
        mel_extractor=mel_extractor,
        manifest=manifest,
        device=device,
        log_wandb=args.wandb,
        log_tb=not args.no_tb,
    )

    start_step = 0
    if args.resume:
        log.info("Resuming from %s", args.resume)
        trainer.load_checkpoint(Path(args.resume))
        # ``step`` field in the checkpoint = m at save-time = the iteration
        # JUST COMPLETED before ckpt_every triggered the save.  Resume at
        # the next iteration so we don't re-do work.
        try:
            payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            start_step = int(payload.get("step", 0)) + 1
        except Exception as e:
            log.warning("Could not extract step from checkpoint: %s — restarting from 0", e)
        log.info("  resume start_step=%d (of %d total)", start_step, cfg.train.M)

    trainer.fit(start_step=start_step)


if __name__ == "__main__":
    main()
