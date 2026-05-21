"""Differentiable F5-TTS sampling wrapper for cycle-consistent test-time training.

The upstream :func:`f5_tts.model.CFM.sample` runs under :func:`torch.no_grad`
and uses :func:`torchdiffeq.odeint` (which does not support gradients with
respect to model parameters through the integration).  For our inner-loop
optimisation we need:

  1. **Autograd through the ODE solver** so the gradient of cycle losses with
     respect to LoRA adapter parameters is well-defined.
  2. **Functional parameter substitution**: each inner-loop step uses a new
     ``θ_k`` value for the adapter weights, derived from ``θ_{k-1}`` via the
     learned optimizer ``ψ``.  We must run the DiT with these overridden
     values without mutating its in-memory parameter buffers (which would
     break autograd through ``ψ``).

We re-implement the Euler integrator in plain PyTorch and route DiT forward
calls through :func:`torch.func.functional_call` so we can pass per-step
overrides for the LoRA ``A``, ``B`` parameters.

A second mode — :meth:`generate_eval` — runs without overrides, with full NFE,
in :func:`torch.no_grad`, and matches the behaviour of the upstream API as
closely as possible.  This is what baseline and evaluation scripts call.

Cycle-gradient truncation
-------------------------
``Risk Mitigation #5`` of the plan and Section 5 of CycleAdapt-TTS handle the
issue of unbounded second-order gradients through a double-generation.  When
``truncate_cycle_grad=True`` we ``.detach()`` the audio prompt going into the
second F5-TTS call: cycle losses still receive gradient signal via the
*second* call's adapter parameters, but they no longer recompute the first
generation when backwarding.  This roughly halves backward time and memory.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from f5_tts.model.modules import get_vocos_mel_spectrogram
from f5_tts.model.utils import (
    convert_char_to_pinyin,
    get_epss_timesteps,
    get_tokenizer,
    lens_to_mask,
)


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def _pad_ref_text(ref_text: str) -> str:
    """Match the upstream API: ensure ref text ends with a sentence terminator."""
    if not (ref_text.endswith(". ") or ref_text.endswith("。")):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "
    return ref_text


def tokens_for(ref_text: str, gen_text: str) -> List[str]:
    """Convert (ref_text + gen_text) to the pinyin-mixed token list F5-TTS expects."""
    ref_text = _pad_ref_text(ref_text)
    combined = [ref_text + gen_text]
    return convert_char_to_pinyin(combined)  # list of list-of-tokens


def text_to_idx(text_tokens: List[List[str]], vocab_char_map: Dict[str, int]) -> torch.Tensor:
    """Map token list to indices; -1 for OOV (will be filtered to padding inside DiT)."""
    out: List[torch.Tensor] = []
    for sent in text_tokens:
        ids = [vocab_char_map.get(t, -1) for t in sent]
        out.append(torch.tensor(ids, dtype=torch.long))
    from torch.nn.utils.rnn import pad_sequence
    return pad_sequence(out, padding_value=-1, batch_first=True)


# ---------------------------------------------------------------------------
# Duration heuristic (matches F5-TTS infer_batch_process logic)
# ---------------------------------------------------------------------------

def estimate_duration_frames(
    ref_wav_24k: torch.Tensor,
    ref_text: str,
    gen_text: str,
    hop_length: int = 256,
    speed: float = 1.0,
) -> int:
    ref_text = _pad_ref_text(ref_text)
    ref_audio_len = ref_wav_24k.shape[-1] // hop_length
    ref_text_len = max(1, len(ref_text.encode("utf-8")))
    gen_text_len = len(gen_text.encode("utf-8"))
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
    return int(duration)


# ---------------------------------------------------------------------------
# Differentiable Euler-step ODE integration with optional functional overrides
# ---------------------------------------------------------------------------

import contextlib


@contextlib.contextmanager
def _swap_lora_params(transformer: nn.Module, overrides: Dict[str, torch.Tensor]):
    """Temporarily replace named LoRA parameters in ``transformer`` with the
    override tensors so that autograd through them works.

    Unlike :func:`torch.func.functional_call`, this approach mutates the
    module's ``_parameters`` dict in-place, which preserves the autograd
    relationship between the override tensor and the module's forward output.
    The original parameters are restored on context exit.
    """
    # Resolve each dotted name to (parent_module, leaf_name) and current Parameter.
    swap_records: list[tuple[nn.Module, str, Any]] = []
    for name, override_tensor in overrides.items():
        parts = name.split(".")
        cur: nn.Module = transformer
        for part in parts[:-1]:
            if part.isdigit():
                cur = cur[int(part)]
            else:
                cur = getattr(cur, part)
        leaf_name = parts[-1]
        original = cur._parameters.get(leaf_name)
        if original is None:
            raise KeyError(f"LoRA override target not found: {name}")
        swap_records.append((cur, leaf_name, original))
        # Direct dict assignment bypasses nn.Parameter type-check and preserves
        # the tensor's autograd connection.
        cur._parameters[leaf_name] = override_tensor  # type: ignore[assignment]
    try:
        yield
    finally:
        for cur, leaf_name, original in swap_records:
            cur._parameters[leaf_name] = original


def _euler_solve(
    transformer: nn.Module,
    y0: torch.Tensor,
    text_idx: torch.Tensor,
    step_cond: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    cfg_strength: float,
    overrides: Optional[Dict[str, torch.Tensor]] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Manual Euler solver with optional LoRA parameter overrides."""
    y = y0
    n_steps = timesteps.shape[0] - 1

    def _one_step(y, t, dt):
        if cfg_strength < 1e-5:
            pred = transformer(
                x=y, cond=step_cond, text=text_idx, time=t, mask=mask,
                cfg_infer=False, cache=False,
            )
        else:
            pred_cfg = transformer(
                x=y, cond=step_cond, text=text_idx, time=t, mask=mask,
                cfg_infer=True, cache=False,
            )
            pred_cond, pred_uncond = torch.chunk(pred_cfg, 2, dim=0)
            pred = pred_cond + (pred_cond - pred_uncond) * cfg_strength
        return y + dt * pred

    if overrides:
        with _swap_lora_params(transformer, overrides):
            for k in range(n_steps):
                t = timesteps[k]
                dt = timesteps[k + 1] - timesteps[k]
                y = _one_step(y, t, dt)
    else:
        for k in range(n_steps):
            t = timesteps[k]
            dt = timesteps[k + 1] - timesteps[k]
            y = _one_step(y, t, dt)

    return y


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

@dataclass
class GenOutput:
    wave: torch.Tensor          # [B, T] @ 24kHz
    mel: torch.Tensor           # [B, n_mel, frames] (only the *generated* part)
    ref_audio_frames: int       # number of mel frames of the ref-audio prefix


class F5CycleWrapper(nn.Module):
    """High-level wrapper around an F5-TTS :class:`CFM` model + Vocos vocoder.

    Exposes two main entry points:

      * :meth:`generate_diff` — autograd-tracked, supports parameter overrides
        and reduced NFE.  Used inside the meta-training inner loop and for
        the cycle reconstruction step.

      * :meth:`generate_eval` — no-grad, full quality, identical numerics
        to the upstream :func:`infer_process`.  Used for baselines / eval.

    Parameters
    ----------
    cfm : f5_tts.model.CFM
        A loaded F5-TTS CFM module (use :func:`load_f5tts_model` helper).
    vocoder : nn.Module
        Vocos vocoder loaded via :func:`f5_tts.infer.utils_infer.load_vocoder`.
    vocab_char_map : dict
        Token → index dictionary from :func:`get_tokenizer`.
    """

    def __init__(self, cfm: nn.Module, vocoder: nn.Module, vocab_char_map: Dict[str, int]):
        super().__init__()
        self.cfm = cfm
        self.vocoder = vocoder
        self.vocab_char_map = vocab_char_map
        self.dit = cfm.transformer
        self.hop_length = cfm.mel_spec.hop_length
        self.n_mel = cfm.mel_spec.n_mel_channels
        self.sample_rate = cfm.mel_spec.target_sample_rate

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_cond_and_text(
        self,
        ref_wav_24k: torch.Tensor,
        ref_text: str,
        gen_text: str,
        duration: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Returns (step_cond, text_idx, ref_audio_frames, total_duration_frames).

        * ``step_cond``  : [1, total_dur, n_mel]  (audio prompt mel + zeros)
        * ``text_idx``   : [1, n_tokens]
        """
        if ref_wav_24k.ndim == 1:
            ref_wav_24k = ref_wav_24k.unsqueeze(0)

        device = next(self.dit.parameters()).device
        dtype = next(self.dit.parameters()).dtype
        ref_wav_24k = ref_wav_24k.to(device=device)

        # Build ref-audio mel spectrogram.
        cond_mel = self.cfm.mel_spec(ref_wav_24k.float())  # [1, n_mel, frames]
        cond_mel = cond_mel.permute(0, 2, 1).to(dtype)     # [1, frames, n_mel]
        ref_audio_frames = cond_mel.shape[1]

        # Duration estimate.
        if duration is None:
            duration = estimate_duration_frames(ref_wav_24k, ref_text, gen_text, hop_length=self.hop_length)
        duration = max(duration, ref_audio_frames + 1)

        # Pad the cond mel with zeros for the part to be generated.
        step_cond = F.pad(cond_mel, (0, 0, 0, duration - ref_audio_frames), value=0.0)

        # Text tokens.
        text_tokens = tokens_for(ref_text, gen_text)
        text_idx = text_to_idx(text_tokens, self.vocab_char_map).to(device)

        return step_cond, text_idx, ref_audio_frames, duration

    @staticmethod
    def _make_timesteps(nfe: int, *, device: torch.device, dtype: torch.dtype, use_epss: bool = True) -> torch.Tensor:
        """Always build timesteps in fp32, then cast: at NFE=32 the consecutive
        EPSS step deltas (~1e-3) collapse to zero in bf16, which the ODE
        solver rejects with a "t must be strictly increasing" error."""
        if use_epss and nfe <= 32:
            try:
                t = get_epss_timesteps(nfe, device=device, dtype=torch.float32)
            except Exception:
                t = torch.linspace(0.0, 1.0, nfe + 1, device=device, dtype=torch.float32)
        else:
            t = torch.linspace(0.0, 1.0, nfe + 1, device=device, dtype=torch.float32)
        return t  # callers may cast to model dtype after applying sway sampling

    @staticmethod
    def _sway(timesteps: torch.Tensor, coef: float) -> torch.Tensor:
        if coef is None:
            return timesteps
        return timesteps + coef * (torch.cos(torch.pi / 2 * timesteps) - 1 + timesteps)

    # ------------------------------------------------------------------
    # Differentiable generation (used for inner-loop and cycle)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def cache_prompt_mel(self, ref_wav_24k: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Precompute prompt mel once per episode (saves 2×K mel-spec ops in inner loop)."""
        if ref_wav_24k.ndim == 1:
            ref_wav_24k = ref_wav_24k.unsqueeze(0)
        device = next(self.dit.parameters()).device
        dtype = next(self.dit.parameters()).dtype
        ref_wav_24k = ref_wav_24k.to(device=device)
        cond_mel = self.cfm.mel_spec(ref_wav_24k.float()).permute(0, 2, 1).to(dtype)
        return cond_mel, cond_mel.shape[1]

    def _build_cond_from_cached_mel(
        self,
        cond_mel: torch.Tensor,
        ref_frames: int,
        ref_text: str,
        gen_text: str,
        duration: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        device = cond_mel.device
        if duration is None:
            stub = torch.zeros(1, max(ref_frames * self.hop_length, 1), device=device)
            duration = estimate_duration_frames(stub, ref_text, gen_text, hop_length=self.hop_length)
        duration = max(duration, ref_frames + 1)
        step_cond = F.pad(cond_mel, (0, 0, 0, duration - ref_frames), value=0.0)
        text_tokens = tokens_for(ref_text, gen_text)
        text_idx = text_to_idx(text_tokens, self.vocab_char_map).to(device)
        return step_cond, text_idx, ref_frames, duration

    def generate_diff(
        self,
        ref_wav_24k: torch.Tensor,
        ref_text: str,
        gen_text: str,
        *,
        nfe_step: int = 8,
        cfg_strength: float = 1.0,
        sway_sampling_coef: Optional[float] = -1.0,
        overrides: Optional[Dict[str, torch.Tensor]] = None,
        seed: Optional[int] = None,
        duration: Optional[int] = None,
        decode_to_wave: bool = True,
        return_full_mel: bool = False,
        cached_cond_mel: Optional[torch.Tensor] = None,
        cached_ref_frames: Optional[int] = None,
    ) -> GenOutput:
        device = next(self.dit.parameters()).device
        dtype = next(self.dit.parameters()).dtype
        if cached_cond_mel is not None and cached_ref_frames is not None:
            step_cond, text_idx, ref_frames, total_dur = self._build_cond_from_cached_mel(
                cached_cond_mel, cached_ref_frames, ref_text, gen_text, duration=duration,
            )
        else:
            step_cond, text_idx, ref_frames, total_dur = self._build_cond_and_text(
                ref_wav_24k, ref_text, gen_text, duration=duration
            )

        # Initial noise — same logic as upstream CFM.sample (per-batch).
        if seed is not None:
            torch.manual_seed(seed)
        y0 = torch.randn(1, total_dur, self.n_mel, device=device, dtype=dtype)

        # Timesteps are kept in fp32 even when the model is bf16 — see
        # ``_make_timesteps`` for the rationale.  Cast to model dtype only
        # AFTER sway sampling has been applied (sway is computed in fp32).
        timesteps = self._make_timesteps(nfe_step, device=device, dtype=dtype, use_epss=True)
        if sway_sampling_coef is not None:
            timesteps = self._sway(timesteps, sway_sampling_coef)
        timesteps = timesteps.to(dtype)

        # Clear the DiT internal text cache so that an override-call doesn't
        # inadvertently re-use a cached text embedding from a previous call.
        self.dit.clear_cache()

        # Solve.
        sampled_mel = _euler_solve(
            self.dit,
            y0=y0,
            text_idx=text_idx,
            step_cond=step_cond,
            timesteps=timesteps,
            cfg_strength=cfg_strength,
            overrides=overrides,
            mask=None,
        )
        # Replace ref-audio frames with the original ref mel (cond_mask logic).
        sampled_mel = torch.cat([step_cond[:, :ref_frames], sampled_mel[:, ref_frames:]], dim=1)

        gen_mel = sampled_mel[:, ref_frames:]  # [1, gen_frames, n_mel]

        if decode_to_wave:
            # NOTE: Vocos's official ``decode`` method is wrapped in
            # ``@torch.inference_mode()`` which strips autograd metadata.  We
            # call the backbone + head explicitly to preserve gradients for
            # cycle losses applied on the waveform.
            mel_in = gen_mel.permute(0, 2, 1).float()  # [B, n_mel, frames]
            voc_x = self.vocoder.backbone(mel_in)
            wave = self.vocoder.head(voc_x)
            if return_full_mel:
                return GenOutput(wave=wave, mel=sampled_mel.permute(0, 2, 1), ref_audio_frames=ref_frames)
            return GenOutput(wave=wave, mel=gen_mel.permute(0, 2, 1), ref_audio_frames=ref_frames)
        else:
            return GenOutput(
                wave=torch.empty(0),
                mel=(sampled_mel if return_full_mel else gen_mel).permute(0, 2, 1),
                ref_audio_frames=ref_frames,
            )

    # ------------------------------------------------------------------
    # No-grad evaluation generation (full quality)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_eval(
        self,
        ref_wav_24k: torch.Tensor,
        ref_text: str,
        gen_text: str,
        *,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        seed: int = 42,
        target_rms: float = 0.1,
    ) -> GenOutput:
        """High-NFE generation for evaluation.

        We re-implement the ODE solver here (rather than calling ``cfm.sample``)
        because the upstream implementation builds the timesteps in
        ``step_cond.dtype``, which is bf16 in our setup.  At NFE=32 the EPSS
        delta-t values (~1e-3) collapse to zero in bf16, breaking the
        ``torchdiffeq.odeint`` monotonicity check.  Our solver keeps t in fp32.
        """
        device = next(self.dit.parameters()).device
        dtype = next(self.dit.parameters()).dtype

        if ref_wav_24k.ndim == 1:
            ref_wav_24k = ref_wav_24k.unsqueeze(0)
        ref_wav_24k = ref_wav_24k.to(device=device)

        # RMS normalize (matches infer_batch_process)
        rms = torch.sqrt(torch.mean(ref_wav_24k.float() ** 2))
        if rms < target_rms and rms > 0:
            ref_wav_24k = ref_wav_24k * target_rms / rms

        # Build the conditioning mel + text exactly as ``generate_diff`` does.
        step_cond, text_idx, ref_frames, total_dur = self._build_cond_and_text(
            ref_wav_24k.squeeze(0), ref_text, gen_text
        )

        if seed is not None:
            torch.manual_seed(seed)
        y0 = torch.randn(1, total_dur, self.n_mel, device=device, dtype=dtype)

        timesteps = self._make_timesteps(nfe_step, device=device, dtype=dtype, use_epss=True)
        if sway_sampling_coef is not None:
            timesteps = self._sway(timesteps, sway_sampling_coef)
        timesteps = timesteps.to(dtype)

        self.dit.clear_cache()
        sampled_mel = _euler_solve(
            self.dit,
            y0=y0,
            text_idx=text_idx,
            step_cond=step_cond,
            timesteps=timesteps,
            cfg_strength=cfg_strength,
            overrides=None,
            mask=None,
        )
        sampled_mel = torch.cat([step_cond[:, :ref_frames], sampled_mel[:, ref_frames:]], dim=1)
        gen_mel = sampled_mel[:, ref_frames:]  # [1, frames, n_mel]

        # Vocos: bypass ``decode`` (inference_mode decorator); call backbone+head.
        mel_in = gen_mel.permute(0, 2, 1).float()
        wave = self.vocoder.head(self.vocoder.backbone(mel_in))

        # Renormalise the gen wave so it sits near the original RMS.
        if rms < target_rms and rms > 0:
            wave = wave * rms / target_rms

        return GenOutput(wave=wave, mel=mel_in, ref_audio_frames=ref_frames)


# ---------------------------------------------------------------------------
# Loader helper (mirrors f5_tts.api.F5TTS but exposes the CFM directly)
# ---------------------------------------------------------------------------

def load_f5tts_model(
    *,
    device: str = "cuda",
    model_name: str = "F5TTS_v1_Base",
    bf16: bool = True,
) -> Tuple[nn.Module, nn.Module, Dict[str, int]]:
    """Load the F5-TTS v1 Base model + Vocos vocoder.  Returns (cfm, vocoder, vocab_char_map)."""
    from cached_path import cached_path
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from f5_tts.infer.utils_infer import load_model as _load_model, load_vocoder

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    vocoder = load_vocoder(mel_spec_type, False, None, device, None)

    ckpt_file = str(cached_path(f"hf://SWivid/F5-TTS/{model_name}/model_1250000.safetensors"))
    cfm = _load_model(model_cls, model_arc, ckpt_file, mel_spec_type, "", "euler", True, device)
    cfm.eval()
    for p in cfm.parameters():
        p.requires_grad = False
    if bf16:
        cfm.transformer.to(dtype=torch.bfloat16)
        # Vocos stays in fp32 for waveform fidelity; the cost is small.

    vocab_path = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    vocab_char_map, _ = get_tokenizer(vocab_path, "custom")

    return cfm, vocoder, vocab_char_map
