# Stability fixes & data FAQ

## Root cause of collapsed inference (trace)

1. **θ₀ in the autograd graph during training**  
   The inner loop used `iaa.init_params[i]` directly as `cur_A`, so inner/outer
   backprop wrote gradients into θ₀ *before* Reptile. That corrupted meta-init.

2. **Reptile + large ψ steps**  
   `output_scale=1e-2` and `reptile_beta=0.3` let **B** grow far from 0.  
   LoRA output `(α/r)·xAᵀBᵀ` then dominated or cancelled the frozen backbone →
   near-silent wavs (peak ~0.06 vs ~0.61 for vanilla F5).

3. **NaN from ~iter 1305**  
   Training continued saving checkpoints with invalid loss; not a data issue.

## Principled fixes (v2)

| Change | Purpose |
|--------|---------|
| Inner loop starts from `init_params.detach().clone()` | θ₀ not in inner graph |
| `ψ output_scale` 1e-4, `max_psi_update_norm` | Small TTT steps |
| `reptile_beta` 0.1, `init_anchor_strength` 0.05 | Slow θ₀ meta-update + pull to pristine |
| Clamp ‖A‖, ‖B‖, ‖A‖_F·‖B‖_F per layer | Adapter can't swamp backbone |
| Skip non-finite `L_outer` iters | No NaN checkpoints |
| Clamp θ₀ on load + `reset_from_init()` at inference | Safe deploy |

## Do we need more data?

**No — not as the first lever.** Current training data:

- LibriTTS-R (EN): multi-speaker, ~247 speakers in manifest  
- AISHELL-3 (ZH): ~174 speakers  
- **~96k training rows** — standard scale for meta-learning TTS adapters  

Collapse happened with **stable optimization**, not from dataset size. More languages
(ES/FR/…) help **zero-shot eval diversity**, not fixing silent output.

Recommended order:

1. Re-train with v2 stability (`--run-name cycleadapt_v2`, M≥800).  
2. Validate with `scripts/trace_lora_collapse.py` (peak amp should be ~0.3–0.8).  
3. Only then add data (e.g. MLS snippets) if zero-shot SIM-o plateaus.

## Re-train command

```bash
cd /home/ubuntu/CYCLE_TTS && source scripts/env.sh
nohup python -u scripts/05_meta_train.py \
  --M 800 --B 4 --K 2 --run-name cycleadapt_v2 \
  > logs/runs/cycleadapt_v2.log 2>&1 &
```

Smoke after ~50 iters:

```bash
python scripts/trace_lora_collapse.py checkpoints/cycleadapt_v2/step000050.pt
```
