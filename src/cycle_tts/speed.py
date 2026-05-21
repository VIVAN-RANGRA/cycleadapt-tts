"""Training/inference speed helpers (no epoch reduction)."""
from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def apply_global_speed_flags() -> None:
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def compile_dit(dit: torch.nn.Module, *, warmup: bool = True) -> torch.nn.Module:
    """``torch.compile`` the DiT backbone (dynamic shapes; first step is slow)."""
    if getattr(dit, "_cycle_compile_wrapped", False):
        return dit
    log.info("torch.compile(DiT) mode=reduce-overhead …")
    compiled = torch.compile(dit, mode="reduce-overhead", fullgraph=False)
    dit._cycle_compile_wrapped = True  # type: ignore[attr-defined]
    if warmup and torch.cuda.is_available():
        log.info("DiT compile warmup skipped (shapes vary per episode)")
    return compiled


def compile_small_modules(*modules: torch.nn.Module) -> None:
    for m in modules:
        if m is None or getattr(m, "_cycle_compile_wrapped", False):
            continue
        try:
            compiled = torch.compile(m, mode="reduce-overhead", fullgraph=True)
            # Copy compile marker onto original module's class path — caller must reassign.
            m._compiled_ref = compiled  # type: ignore[attr-defined]
            m._cycle_compile_wrapped = True  # type: ignore[attr-defined]
        except Exception as e:
            log.warning("compile skipped for %s: %s", m.__class__.__name__, e)
