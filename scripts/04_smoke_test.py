"""Stage 2b — end-to-end smoke test.

Verifies that:
  1. The full system loads (F5-TTS + IAA + ψ + φ + extractors).
  2. We can sample an episode from the train manifest.
  3. A single inner-loop step produces finite losses with autograd-tracked
     gradients flowing to ψ and φ.
  4. The outer-loop generation runs and produces a usable waveform.
  5. Memory stays under 60 GB for batch=1.

Prints peak GPU memory, per-step timing, and a small audio sample to
``results/audio_samples/smoke/``.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import soundfile as sf
import torch

from cycle_tts.config import CycleAdaptConfig
from cycle_tts.data import ManifestIndex
from cycle_tts.episode_sampler import EpisodeSampler
from cycle_tts.f5_wrapper import F5CycleWrapper, load_f5tts_model
from cycle_tts.feature_extractors import (
    F0Extractor,
    MelSpecExtractor,
    WavLMSpeakerEncoder,
)
from cycle_tts.iaa import IdentityAlignmentAdapter
from cycle_tts.loss_weighter import LossWeighter
from cycle_tts.losses import compute_loss_bundle, outer_loss
from cycle_tts.meta_optimizer import LearnedOptimizer
from cycle_tts.meta_trainer import _inner_loop, _lora_param_name_to_path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("smoke")

CYCLE_ROOT = Path(os.environ.get("CYCLE_TTS_ROOT", "/home/ubuntu/CYCLE_TTS"))


def main() -> None:
    cfg = CycleAdaptConfig(run_name="smoke_test")
    device = "cuda"

    log.info("Loading F5-TTS v1 Base ...")
    t0 = time.time()
    cfm, vocoder, vocab_char_map = load_f5tts_model(device=device, bf16=True)
    f5 = F5CycleWrapper(cfm, vocoder, vocab_char_map).to(device)
    log.info("  F5-TTS loaded in %.1fs", time.time() - t0)

    log.info("Building IAA ...")
    iaa = IdentityAlignmentAdapter(cfg.iaa, cfm.transformer).to(device)
    log.info("  IAA params=%d injection_records=%d", iaa.n_adapter_params(), len(iaa.records))

    log.info("Building ψ (learned optimizer) ...")
    psi = LearnedOptimizer(cfg.psi).to(device)
    log.info("  ψ params=%d", psi.num_params())

    log.info("Building φ (loss weighter) ...")
    phi = LossWeighter(cfg.phi).to(device)
    log.info("  φ params=%d", sum(p.numel() for p in phi.parameters()))

    log.info("Loading feature extractors ...")
    spk_encoder = WavLMSpeakerEncoder(device=device)
    f0_extractor = F0Extractor(device=device)
    mel_extractor = MelSpecExtractor(cfm.mel_spec)

    log.info("Loading manifests ...")
    manifest = ManifestIndex.from_jsonl_files([
        CYCLE_ROOT / "data" / "manifests" / "libritts_r_train.jsonl",
        CYCLE_ROOT / "data" / "manifests" / "aishell3_train.jsonl",
    ])
    log.info("  total rows=%d  EN speakers=%d  ZH speakers=%d",
             len(manifest.rows),
             manifest.n_speakers("en"),
             manifest.n_speakers("zh"))

    sampler = EpisodeSampler(cfg, manifest, seed=0)
    episode = sampler.sample()
    log.info("Sampled episode: speaker=%s  %s->%s  t_support=%s  t_query=%s",
             episode.speaker_id, episode.lang_l1, episode.lang_l2,
             episode.t_support[:60], episode.t_query[:60])

    log.info("\n=== INNER LOOP STEP ===")
    paths = _lora_param_name_to_path(iaa)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    # MICRO TEST: confirm a single LoRA forward produces grad_fn'd output.
    test_lora = iaa.records[0].lora
    log.info("MICRO LoRA test: A.requires_grad=%s A.dtype=%s base.weight.requires_grad=%s",
             test_lora.A.requires_grad, test_lora.A.dtype, test_lora.base.weight.requires_grad)
    x_test = torch.randn(1, 8, test_lora.base.in_features, device=device,
                          dtype=test_lora.base.weight.dtype)
    y_test = test_lora(x_test)
    log.info("  LoRA(x_random).requires_grad=%s grad_fn=%s",
             y_test.requires_grad,
             type(y_test.grad_fn).__name__ if y_test.grad_fn else "None")

    # Sanity check 1: confirm override path names match named_parameters().
    nps = {n for n, _ in cfm.transformer.named_parameters()}
    from cycle_tts.meta_trainer import _make_override
    n_rec = len(iaa.records)
    sample_path = f"transformer_blocks.{iaa.records[0].block_idx}.{iaa.records[0].submodule_path}.A"
    log.info("Override path sample: %s  matches param? %s", sample_path, sample_path in nps)

    # Sanity check 2: build a NONZERO override (large B) and verify the wave differs.
    lora_dtype = iaa.records[0].lora.A.dtype
    test_A = [torch.randn_like(iaa.init_params[i], dtype=lora_dtype) * 0.1 for i in range(n_rec)]
    test_B = [torch.randn_like(iaa.init_params[n_rec + i], dtype=lora_dtype) * 0.5 for i in range(n_rec)]
    # Mark as requires_grad to mimic the real inner-loop tensor properties.
    for t in test_A + test_B:
        t.requires_grad_(True)
    test_override = _make_override(iaa, test_A, test_B, paths)
    out1 = f5.generate_diff(episode.x_wav_24k, episode.t_prime, episode.t_support,
                             nfe_step=4, cfg_strength=2.0, overrides=test_override, seed=0)
    out2 = f5.generate_diff(episode.x_wav_24k, episode.t_prime, episode.t_support,
                             nfe_step=4, cfg_strength=2.0, overrides=None, seed=0)
    diff = (out1.wave - out2.wave).abs().mean().item()
    log.info("Override (random A,B) sanity: |with - without| = %.6f, out1.wave req_grad=%s grad_fn=%s",
             diff, out1.wave.requires_grad,
             type(out1.wave.grad_fn).__name__ if out1.wave.grad_fn else "None")

    inner = _inner_loop(
        episode=episode,
        iaa=iaa,
        psi=psi,
        phi=phi,
        f5=f5,
        spk_encoder=spk_encoder,
        f0_extractor=f0_extractor,
        mel_extractor=mel_extractor,
        cfg=cfg,
        paths=paths,
        K=cfg.cycle.K_train,
        cycle_grad_truncate=True,
        log_progress=True,
    )
    dt_inner = time.time() - t0
    log.info("Inner loop done in %.2fs.  inner_losses=%s", dt_inner, [f"{x:.3f}" for x in inner.inner_step_losses])
    log.info("Peak GPU memory after inner: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)

    log.info("Final cur_A[0] requires_grad=%s grad_fn=%s",
             inner.final_A[0].requires_grad,
             type(inner.final_A[0].grad_fn).__name__ if inner.final_A[0].grad_fn else "None")

    log.info("\n=== OUTER LOOP ===")
    t0 = time.time()
    gen_q = f5.generate_diff(
        episode.x_wav_24k,
        episode.t_prime,
        episode.t_query,
        nfe_step=cfg.cycle.nfe_outer,
        cfg_strength=cfg.cycle.cfg_strength_outer,
        overrides=inner.last_override,
        seed=42,
    )
    L_outer, diag = outer_loss(
        episode.x_wav_24k.to(device),
        gen_q.wave,
        spk_encoder=spk_encoder,
        f0_extractor=f0_extractor,
        lambda_f0=cfg.cycle.outer_lambda_f0,
    )
    log.info("Outer gen+loss done in %.2fs.  L_outer=%.4f  (SIM=%.4f F0=%.4f)",
             time.time() - t0, float(L_outer), float(diag["outer_sim"]), float(diag["outer_f0"]))
    log.info("gen_q.wave req_grad=%s grad_fn=%s",
             gen_q.wave.requires_grad,
             type(gen_q.wave.grad_fn).__name__ if gen_q.wave.grad_fn else "None")
    log.info("L_outer req_grad=%s grad_fn=%s",
             L_outer.requires_grad,
             type(L_outer.grad_fn).__name__ if L_outer.grad_fn else "None")

    log.info("\n=== AUTOGRAD CHECK ===")
    t0 = time.time()
    # ψ via L_outer (FOMAML), retain_graph so we can also backprop through the
    # φ-only auxiliary loss recorded by _inner_loop.
    psi_grads_raw = torch.autograd.grad(
        L_outer, list(psi.parameters()), retain_graph=True, allow_unused=True
    )
    psi_grad_norms = [g.norm().item() if g is not None else 0.0 for g in psi_grads_raw]
    # φ via the inner-loop auxiliary loss (returned by _inner_loop).
    phi_grads_raw = torch.autograd.grad(
        inner.last_L_total_for_phi, list(phi.parameters()), allow_unused=True
    )
    phi_grad_norms = [g.norm().item() if g is not None else 0.0 for g in phi_grads_raw]
    log.info("Outer backward in %.2fs.  ψ grad sum-of-norms=%.4f, φ grad sum-of-norms=%.4f",
             time.time() - t0, sum(psi_grad_norms), sum(phi_grad_norms))

    # Save sample
    out_dir = CYCLE_ROOT / "results" / "audio_samples" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "smoke_prompt.wav",
             episode.x_wav_24k.squeeze().cpu().numpy(), 24_000)
    sf.write(out_dir / "smoke_outer_query_gen.wav",
             gen_q.wave.squeeze().detach().float().cpu().numpy(), 24_000)
    log.info("Saved samples -> %s", out_dir)
    log.info("Peak GPU memory final: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
    main()
