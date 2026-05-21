#!/usr/bin/env python
"""Our method — test-time adapt (TTT) on the prompt then generate.

Pipeline per eval item:

  1. Load (frozen) F5-TTS + IAA + ψ + φ from a trained checkpoint.
  2. Reset IAA's leaf params to ``θ_0`` (the meta-learned init).
  3. Run K_test inner-loop adaptation steps on the *prompt only* (using the
     same cycle-consistent objective as training, with the prompt as both
     reference and target context).
  4. With the adapted parameters, generate the target wave at NFE=32.

Writes to::

    results/audio/{run_name}_{ckpt_step}/{pair_id}_{slot:03d}.wav
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

ROOT = Path("/home/ubuntu/CYCLE_TTS")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party_f5" / "src"))

from cycle_tts.config import CycleAdaptConfig  # noqa: E402
from cycle_tts.eval_prompts import EvalItem, load_eval_set  # noqa: E402
from cycle_tts.f5_wrapper import F5CycleWrapper, load_f5tts_model  # noqa: E402
from cycle_tts.feature_extractors import F0Extractor, MelSpecExtractor, WavLMSpeakerEncoder  # noqa: E402
from cycle_tts.iaa import IdentityAlignmentAdapter  # noqa: E402
from cycle_tts.loss_weighter import LossWeighter  # noqa: E402
from cycle_tts.losses import compute_loss_bundle, embed_prompt_24k, speaker_cosine_loss_from_embed  # noqa: E402
from cycle_tts.metrics import EvalRecord  # noqa: E402
from cycle_tts.meta_optimizer import LearnedOptimizer  # noqa: E402
from cycle_tts.lora_stability import bound_psi_updates, clamp_init_params, clamp_lora_lists  # noqa: E402
from cycle_tts.meta_trainer import _lora_param_name_to_path, _make_override  # noqa: E402


def load_prompt_wav_24k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 24_000:
        wav = torchaudio.functional.resample(wav, sr, 24_000)
    return wav.squeeze(0)


def adapt_and_generate(
    *,
    item: EvalItem,
    iaa: IdentityAlignmentAdapter,
    psi: LearnedOptimizer,
    phi: LossWeighter,
    f5: F5CycleWrapper,
    spk_encoder: WavLMSpeakerEncoder,
    f0_extractor: F0Extractor,
    mel_extractor: MelSpecExtractor,
    cfg: CycleAdaptConfig,
    paths,
    K_test: int,
    final_nfe: int,
    final_cfg_strength: float,
    device: str,
    use_adam: bool = False,
    adam_lr: float = 1e-3,
    no_phi: bool = False,
    no_cycle: bool = False,
    lid_only: bool = False,
    rerank_candidates: int = 1,
    rerank_scorer: str = "wavlm",
    rerank_ecapa_weight: float = 0.3,
    rerank_asr_weight: float = 0.05,
    rerank_asr_topk: int = 2,
    ecapa_scorer=None,
    asr_scorer=None,
) -> torch.Tensor:
    """K_test inner steps on the prompt, then a single eval generation."""
    dtype = next(iaa.dit.parameters()).dtype
    n_rec = len(iaa.records)
    cur_A = [iaa.init_params[i].detach().to(dtype) for i in range(n_rec)]
    cur_B = [iaa.init_params[n_rec + i].detach().to(dtype) for i in range(n_rec)]
    for t in cur_A + cur_B:
        t.requires_grad_(True)

    prompt_wav = load_prompt_wav_24k(item.prompt_wav).to(device)
    prompt_wav_cpu = prompt_wav.detach().float().cpu()
    with torch.no_grad():
        cached_cond_mel, cached_ref_frames = f5.cache_prompt_mel(prompt_wav)
        e_prompt = embed_prompt_24k(prompt_wav.unsqueeze(0), spk_encoder)
        e_prompt_ecapa = None
        if ecapa_scorer is not None and "ecapa" in rerank_scorer:
            e_prompt_ecapa = ecapa_scorer.embed(prompt_wav.detach().float().cpu())

    adam_opt = None
    if use_adam:
        # We need leaf params for Adam; cur_A/cur_B are already leaves with
        # requires_grad=True so we just wrap them.
        adam_opt = torch.optim.Adam(cur_A + cur_B, lr=adam_lr)

    for k in range(K_test):
        # Use the prompt itself as both x and the L2 generation in
        # ``support`` text (a self-cycle).  ``t_query`` is taken to be the
        # actual target generation text (same as final).
        override = _make_override(iaa, cur_A, cur_B, paths)
        gen_fwd = f5.generate_diff(
            prompt_wav,
            item.prompt_text,
            item.gen_text,
            nfe_step=cfg.cycle.nfe_forward_inner,
            cfg_strength=cfg.cycle.cfg_strength_inner,
            sway_sampling_coef=-1.0,
            overrides=override,
            seed=k * 31 + 7,
            cached_cond_mel=cached_cond_mel,
            cached_ref_frames=cached_ref_frames,
        )
        y_hat = gen_fwd.wave

        if lid_only or no_cycle:
            y_hat_cycle = y_hat
        else:
            y_hat_for_cycle = y_hat.detach() if cfg.cycle.truncate_cycle_grad else y_hat
            gen_cyc = f5.generate_diff(
                y_hat_for_cycle.squeeze(0),
                item.gen_text,
                item.prompt_text,
                nfe_step=cfg.cycle.nfe_cycle_inner,
                cfg_strength=cfg.cycle.cfg_strength_inner,
                sway_sampling_coef=-1.0,
                overrides=override,
                seed=k * 17 + 3,
            )
            y_hat_cycle = gen_cyc.wave

        bundle = compute_loss_bundle(
            x_prompt_24k=prompt_wav.unsqueeze(0),
            y_hat_24k=y_hat,
            y_hat_cycle_24k=y_hat_cycle,
            spk_encoder=spk_encoder,
            f0_extractor=f0_extractor,
            mel_extractor=mel_extractor,
            e_prompt=e_prompt,
            id_only=lid_only,
        )
        lam = cfg.cycle.inner_loss_lambda
        if lid_only or no_cycle:
            mask = torch.tensor(
                [0.0, 0.0, 0.0, lam["id"], 0.0],
                device=device, dtype=dtype,
            )
        else:
            mask = torch.tensor(
                [lam["spk"], lam["spec"], lam["f0"], lam["id"], lam["intel"]],
                device=device, dtype=dtype,
            )
        if no_phi:
            # Uniform 1/5 weights → after mask, equal weight to active losses.
            w = torch.full((cfg.phi.n_losses,), 1.0 / cfg.phi.n_losses,
                            device=device, dtype=dtype)
        else:
            L1_idx = cfg.lang_idx.get(item.L1, 0)
            L2_idx = cfg.lang_idx.get(item.L2, 0)
            with torch.no_grad():
                w = phi(bundle.to_vector().detach(), L1_idx, L2_idx, active_mask=mask > 0).to(dtype)
        L_total = bundle.weighted_sum(w, mask=mask)

        params = cur_A + cur_B

        if use_adam:
            # Vanilla Adam TTT.  Backprop into the leaves; step the optimizer.
            adam_opt.zero_grad(set_to_none=True)
            L_total.backward()
            adam_opt.step()
            # Re-mark as leaves with grad so the next iteration's autograd
            # works (Adam's in-place mutation already keeps them as leaves).
        else:
            grads = torch.autograd.grad(
                L_total, params, create_graph=False, retain_graph=False, allow_unused=True
            )
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
            grads_A, grads_B = grads[:n_rec], grads[n_rec:]
            updates_A = psi.compute_update([g.detach() for g in grads_A], loss=L_total.detach(), step=k)
            updates_B = psi.compute_update([g.detach() for g in grads_B], loss=L_total.detach(), step=k)
            if cfg.cycle.residual_grad_lr > 0:
                lr = cfg.cycle.residual_grad_lr
                updates_A = [u + lr * g.detach().to(dtype=u.dtype) for u, g in zip(updates_A, grads_A)]
                updates_B = [u + lr * g.detach().to(dtype=u.dtype) for u, g in zip(updates_B, grads_B)]
            updates_A = bound_psi_updates(updates_A, cfg.stab)
            updates_B = bound_psi_updates(updates_B, cfg.stab)
            new_A = [(a - u).detach() for a, u in zip(cur_A, updates_A)]
            new_B = [(b - u).detach() for b, u in zip(cur_B, updates_B)]
            clamp_lora_lists(new_A, new_B, cfg.stab, cfg.iaa)
            cur_A = [t.requires_grad_(True) for t in new_A]
            cur_B = [t.requires_grad_(True) for t in new_B]

    # Final generation with adapted params at high NFE.
    final_override = _make_override(iaa, cur_A, cur_B, paths)
    candidates = []
    with torch.inference_mode():
        for c in range(max(1, rerank_candidates)):
            out = f5.generate_diff(
                prompt_wav,
                item.prompt_text,
                item.gen_text,
                nfe_step=final_nfe,
                cfg_strength=final_cfg_strength,
                sway_sampling_coef=-1.0,
                overrides=final_override,
                seed=42 + 997 * c,
                cached_cond_mel=cached_cond_mel,
                cached_ref_frames=cached_ref_frames,
            )
            wave_cpu = out.wave.squeeze(0).detach().float().cpu()
            if rerank_candidates <= 1:
                return wave_cpu
                break
            # Identity-aware decoding: choose the sampled trajectory whose
            # output is closest to the prompt in off-the-shelf verifier space.
            # This is prompt-only selection: it never sees the target waveform.
            wavlm_score = -float(speaker_cosine_loss_from_embed(e_prompt, out.wave, spk_encoder).detach())
            score = wavlm_score
            if e_prompt_ecapa is not None and ecapa_scorer is not None:
                e_gen_ecapa = ecapa_scorer.embed(wave_cpu)
                ecapa_score = float(F.cosine_similarity(e_prompt_ecapa, e_gen_ecapa, dim=-1).item())
                score = wavlm_score + rerank_ecapa_weight * ecapa_score
            candidates.append({"score": score, "wave": wave_cpu, "candidate": c})
            del out
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    if not candidates:
        raise RuntimeError("No final candidates were generated")

    best = max(candidates, key=lambda x: x["score"])
    if asr_scorer is not None and "asr" in rerank_scorer:
        # ASR is the heaviest verifier.  Apply it only to the top candidates
        # from the cheap speaker reranker, preserving the ASR signal without
        # running Whisper 8x per item on a crowded synthesis process.
        topk = max(1, min(int(rerank_asr_topk), len(candidates)))
        best_score = float("-inf")
        for cand in sorted(candidates, key=lambda x: x["score"], reverse=True)[:topk]:
            rec = EvalRecord(
                item_id=f"{item.pair_id}_{item.slot:03d}",
                pair_id=item.pair_id,
                pair_class=item.pair_class,
                L1=item.L1,
                L2=item.L2,
                prompt_wav=prompt_wav_cpu,
                gen_wav=cand["wave"],
                prompt_text=item.prompt_text,
                gen_text=item.gen_text,
                method="rerank",
            )
            try:
                asr_err = asr_scorer.score(rec)
            except Exception as exc:
                logging.getLogger("ours").warning(
                    "ASR rerank failed for %s candidate %d: %s",
                    rec.item_id, cand["candidate"], exc,
                )
                asr_err = 10.0
            if not torch.isfinite(torch.tensor(asr_err)):
                asr_err = 10.0
            score = cand["score"] - rerank_asr_weight * float(asr_err)
            if score > best_score:
                best_score = score
                best = cand
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return best["wave"].float().cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="",
                   help="path to a meta-trained checkpoint .pt file (omit for random-init B2)")
    p.add_argument("--eval-set", type=str, default=str(ROOT / "results" / "eval_set.jsonl"))
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--K-test", type=int, default=None,
                   help="override cfg.cycle.K_test")
    p.add_argument("--final-nfe", type=int, default=32)
    p.add_argument("--final-cfg-strength", type=float, default=2.0)
    p.add_argument("--rerank-candidates", type=int, default=1,
                   help="generate N final seeds and keep the highest prompt-SIM candidate")
    p.add_argument("--rerank-scorer", choices=["wavlm", "wavlm_ecapa", "wavlm_ecapa_asr"], default="wavlm",
                   help="prompt-only verifier used for candidate selection")
    p.add_argument("--rerank-ecapa-weight", type=float, default=0.3,
                   help="ECAPA weight when --rerank-scorer=wavlm_ecapa")
    p.add_argument("--rerank-asr-weight", type=float, default=0.05,
                   help="ASR error penalty weight when --rerank-scorer=wavlm_ecapa_asr")
    p.add_argument("--rerank-asr-topk", type=int, default=2,
                   help="apply ASR reranking only to the top-k WavLM/ECAPA candidates")
    p.add_argument("--rerank-asr-device", type=str, default="cpu",
                   help="device for ASR reranking; CPU/int8 avoids crowding synthesis on CUDA")
    p.add_argument("--rerank-asr-model-size", type=str, default="large-v3-turbo",
                   help="faster-whisper model used only for ASR reranking")
    p.add_argument("--rerank-asr-compute-type", type=str, default=None,
                   help="optional faster-whisper compute_type override for ASR reranking")
    p.add_argument("--device", type=str, default="cuda")
    # Method-variant flags
    p.add_argument("--no-meta-init", action="store_true",
                   help="B2 variant: ignore checkpoint's iaa.init_params; use random LoRA init")
    p.add_argument("--use-adam", action="store_true",
                   help="B2 variant: replace \u03c8 with vanilla Adam in the inner loop")
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--no-phi", action="store_true",
                   help="ablation: use uniform inner-loop weights instead of \u03c6")
    p.add_argument("--no-cycle", action="store_true",
                   help="ablation A3: drop the cycle losses (only L_id is back-propped)")
    p.add_argument("--id-only-ttt", action="store_true",
                   help="TTT with L_id only (maximize prompt↔gen speaker match)")
    p.add_argument("--method-name", type=str, default="ours",
                   help="label written into summary.json")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate existing WAVs instead of skipping them")
    p.add_argument("--compile", action="store_true",
                   help="experimental torch.compile(DiT); disabled by default because dynamic-shape eval is not numerically stable")
    p.add_argument("--shard-id", type=int, default=0, help="shard index for multi-GPU eval")
    p.add_argument("--num-shards", type=int, default=1, help="total shards (one GPU per shard)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("ours")

    from cycle_tts.speed import apply_global_speed_flags
    apply_global_speed_flags()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CycleAdaptConfig()

    # ---- load models ----
    log.info("Loading F5-TTS …")
    cfm, vocoder, vocab_char_map = load_f5tts_model(device=args.device, bf16=True)
    f5 = F5CycleWrapper(cfm, vocoder, vocab_char_map)
    K_test = cfg.cycle.K_test if args.K_test is None else args.K_test
    if args.compile:
        log.warning(
            "--compile requested but disabled: dynamic-shape DiT compilation "
            "changed generated audio in EMNLP eval. Running eager mode."
        )

    log.info("Building IAA + ψ + φ …")
    iaa = IdentityAlignmentAdapter(cfg.iaa, cfm.transformer)
    iaa.freeze_base()
    psi = LearnedOptimizer(cfg.psi).to(args.device)
    phi = LossWeighter(cfg.phi).to(args.device)
    psi.eval(); phi.eval()
    for p_ in psi.parameters(): p_.requires_grad = False
    for p_ in phi.parameters(): p_.requires_grad = False

    if args.ckpt:
        log.info("Loading checkpoint %s", args.ckpt)
        payload = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        iaa.load_state_dict_iaa_only(payload["iaa"])
        clamp_init_params(iaa.init_params, len(iaa.records), cfg.stab, cfg.iaa)
        iaa.reset_from_init()
        psi.load_state_dict(payload["psi"])
        phi.load_state_dict(payload["phi"])
    else:
        log.info("--ckpt omitted: using random IAA init + random \u03c8/\u03c6 (B2 baseline mode)")

    if args.no_meta_init:
        # Re-randomize the meta-init in-place so we exercise the inner loop with
        # random LoRA weights (B2 baseline).
        log.info("--no-meta-init: re-randomising IAA \u03b8\u2080 with stddev=%g", cfg.iaa.init_std)
        with torch.no_grad():
            for p_ in iaa.init_parameters():
                if p_.dim() >= 2:
                    torch.nn.init.normal_(p_, std=cfg.iaa.init_std)
                else:
                    p_.zero_()

    paths = _lora_param_name_to_path(iaa)

    log.info("Loading feature extractors …")
    spk_encoder = WavLMSpeakerEncoder(device=args.device)
    f0_extractor = F0Extractor(device=args.device)
    mel_extractor = MelSpecExtractor(cfm.mel_spec)
    ecapa_scorer = None
    asr_scorer = None
    if "ecapa" in args.rerank_scorer and args.rerank_candidates > 1:
        from cycle_tts.metrics import ECAPASim
        log.info("Loading ECAPA reranker …")
        ecapa_scorer = ECAPASim(args.device)
    if "asr" in args.rerank_scorer and args.rerank_candidates > 1:
        from cycle_tts.metrics import WhisperWER
        log.info("Loading ASR reranker on %s …", args.rerank_asr_device)
        asr_scorer = WhisperWER(
            args.rerank_asr_device,
            model_size=args.rerank_asr_model_size,
            compute_type=args.rerank_asr_compute_type,
        )

    items = load_eval_set(Path(args.eval_set))
    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
        log.info("Shard %d/%d -> %d items", args.shard_id, args.num_shards, len(items))
    log.info("Eval items: %d  K_test=%d  final_nfe=%d", len(items), K_test, args.final_nfe)

    timings_path = out_dir / "timings.jsonl"
    summary_path = out_dir / "summary.json"
    t_total = time.time()
    skipped = 0
    existing_skipped = 0
    timings = []

    timings_mode = "w" if args.overwrite else ("a" if timings_path.exists() and timings_path.stat().st_size > 0 else "w")
    with timings_path.open(timings_mode) as tf:
        for i, it in enumerate(items):
            out_path = out_dir / f"{it.pair_id}_{it.slot:03d}.wav"
            if out_path.exists() and not args.overwrite:
                existing_skipped += 1
                continue
            if out_path.exists() and args.overwrite:
                out_path.unlink()
            t0 = time.time()
            try:
                wave = adapt_and_generate(
                    item=it,
                    iaa=iaa, psi=psi, phi=phi, f5=f5,
                    spk_encoder=spk_encoder,
                    f0_extractor=f0_extractor,
                    mel_extractor=mel_extractor,
                    cfg=cfg, paths=paths,
                    K_test=K_test,
                    final_nfe=args.final_nfe,
                    final_cfg_strength=args.final_cfg_strength,
                    device=args.device,
                    use_adam=args.use_adam,
                    adam_lr=args.adam_lr,
                    no_phi=args.no_phi,
                    no_cycle=args.no_cycle,
                    lid_only=args.id_only_ttt,
                    rerank_candidates=args.rerank_candidates,
                    rerank_scorer=args.rerank_scorer,
                    rerank_ecapa_weight=args.rerank_ecapa_weight,
                    rerank_asr_weight=args.rerank_asr_weight,
                    rerank_asr_topk=args.rerank_asr_topk,
                    ecapa_scorer=ecapa_scorer,
                    asr_scorer=asr_scorer,
                )
            except Exception as e:
                log.warning("Skipping %s (gen failed): %s", out_path.name, e)
                import traceback; traceback.print_exc()
                skipped += 1
                continue
            sf.write(out_path, wave.numpy(), 24_000)
            dt = time.time() - t0
            rec = {
                "item": out_path.name,
                "pair_id": it.pair_id,
                "L1": it.L1, "L2": it.L2,
                "gen_sec": float(wave.shape[-1] / 24_000),
                "elapsed_sec": dt,
                "rtf": dt / max(wave.shape[-1] / 24_000, 1e-6),
            }
            tf.write(json.dumps(rec) + "\n")
            tf.flush()
            timings.append(rec)
            if (i + 1) % 10 == 0:
                log.info("[%d/%d]  %s  RTF=%.2f  elapsed=%.0fs",
                         i + 1, len(items), out_path.name, rec["rtf"], time.time() - t_total)

    t_total = time.time() - t_total
    summary = {
        "method": args.method_name,
        "ckpt": args.ckpt,
        "n_items": len(items),
        "n_generated": len(timings),
        "n_existing_skipped": existing_skipped,
        "n_skipped": skipped,
        "K_test": K_test,
        "final_nfe": args.final_nfe,
        "final_cfg_strength": args.final_cfg_strength,
        "rerank_candidates": args.rerank_candidates,
        "rerank_scorer": args.rerank_scorer,
        "rerank_ecapa_weight": args.rerank_ecapa_weight,
        "rerank_asr_weight": args.rerank_asr_weight,
        "rerank_asr_topk": args.rerank_asr_topk,
        "rerank_asr_device": args.rerank_asr_device,
        "rerank_asr_model_size": args.rerank_asr_model_size,
        "rerank_asr_compute_type": args.rerank_asr_compute_type,
        "no_meta_init": args.no_meta_init,
        "use_adam": args.use_adam,
        "no_phi": args.no_phi,
        "no_cycle": args.no_cycle,
        "lid_only": args.id_only_ttt,
        "wall_seconds": t_total,
        "mean_rtf": (sum(t["rtf"] for t in timings) / len(timings)) if timings else float("nan"),
        "out_dir": str(out_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("DONE  %d items in %.0fs (%.2f s/item).  Summary -> %s",
             len(items) - skipped, t_total, t_total / max(len(items) - skipped, 1), summary_path)
    if skipped:
        raise SystemExit(f"Generation failed for {skipped}/{len(items)} items; refusing partial result.")


if __name__ == "__main__":
    main()
