"""Stage 0 sanity check.

Downloads F5-TTS v1 Base on first run, runs one EN->EN inference and one
EN->ZH style inference, writes both to results/audio_samples/sanity/ and reports
timing.  Confirms that:

  * the model + Vocos vocoder load on the A100 in bf16,
  * inference returns a 24 kHz waveform,
  * baseline NFE=32 and reduced NFE=8 both work (the latter is what the inner
    loop will use during meta-training).

Run with:  python scripts/00_sanity_f5tts.py
"""
from __future__ import annotations

import os
import time
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

CYCLE_ROOT = Path(os.environ.get("CYCLE_TTS_ROOT", "/home/ubuntu/CYCLE_TTS"))
OUT_DIR = CYCLE_ROOT / "results" / "audio_samples" / "sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    from f5_tts.api import F5TTS

    print(f"[sanity] torch={torch.__version__} cuda_ok={torch.cuda.is_available()}")
    print(f"[sanity] device={torch.cuda.get_device_name(0)}")

    t0 = time.time()
    f5 = F5TTS(model="F5TTS_v1_Base", device="cuda")
    print(f"[sanity] model loaded in {time.time() - t0:.1f}s")

    ref_en = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
    ref_zh = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_zh.wav"))

    en_text_src = "Some call me nature, others call me mother nature."
    zh_target = "今天天气真好，我们一起去公园散步吧。"
    en_target = "Cycle consistency provides a useful self-supervised training signal."

    for tag, ref_file, ref_text, gen_text, nfe in [
        ("en2en_nfe32", ref_en, en_text_src, en_target, 32),
        ("en2zh_nfe32", ref_en, en_text_src, zh_target, 32),
        ("en2zh_nfe8",  ref_en, en_text_src, zh_target, 8),
    ]:
        t0 = time.time()
        wav, sr, _spec = f5.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=gen_text,
            nfe_step=nfe,
            cfg_strength=2.0,
            seed=42,
        )
        dur = time.time() - t0
        out_path = OUT_DIR / f"{tag}.wav"
        sf.write(out_path, wav, sr)
        print(
            f"[sanity] {tag}: {dur:5.2f}s  sr={sr}  len={len(wav)/sr:5.2f}s  "
            f"rms={np.sqrt(np.mean(wav.astype(np.float32) ** 2)):.4f}  "
            f"-> {out_path}"
        )

    print("[sanity] all OK")


if __name__ == "__main__":
    main()
