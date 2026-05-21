"""Central configuration for CycleAdapt-TTS."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("CYCLE_TTS_ROOT", "/home/ubuntu/CYCLE_TTS"))


@dataclass
class IAAConfig:
    """Identity Alignment Adapter (LoRA) config."""

    rank: int = 8
    alpha: float = 16.0  # LoRA scaling = alpha / rank
    init_std: float = 0.02
    # Inject into top-N DiT blocks (depth=22 for F5TTS_v1_Base, so 18..21).
    n_top_blocks: int = 4
    # Which sub-modules to wrap per block.
    # 'qv' = LoRA on to_q + to_v (standard PEFT default, light)
    # 'qkv' = + to_k as well
    # 'qkvo' = + to_out[0]
    # 'qkvo_ada' = + attn_norm.linear (adaLN modulation projection)
    target_modules: str = "qkvo_ada"

    def total_params_estimate(self, dim: int = 1024) -> int:
        """Rough param count: 2 * rank * dim per target matrix * n_targets * n_blocks."""
        per_target = 2 * self.rank * dim
        n_targets = {"qv": 2, "qkv": 3, "qkvo": 4, "qkvo_ada": 4 + 6}[self.target_modules]
        return per_target * n_targets * self.n_top_blocks


@dataclass
class MetaOptConfig:
    """Learned optimizer ψ config."""

    hidden_dim: int = 64
    n_layers: int = 2
    use_log_grad: bool = True
    use_grad_sign: bool = True
    use_loss_input: bool = True
    use_step_input: bool = True
    max_step_pe: int = 16
    output_scale: float = 1e-4  # small ψ steps — was 1e-2 and blew up θ₀


@dataclass
class LossWeighterConfig:
    """Learned loss-weighter φ config."""

    hidden_dim: int = 32
    n_layers: int = 2
    n_losses: int = 5
    lang_emb_dim: int = 16
    n_languages: int = 8  # en, es, zh, hi, fr, de, ja, ko (only first 3-4 actually used)
    use_softmax: bool = True
    weight_floor: float = 0.02  # min weight after softmax (clipped)


@dataclass
class CycleConfig:
    """Cycle-consistent test-time training config."""

    # NOTE: hyperparameters tuned for a 2-day A100 budget — see scripts/05_meta_train.py
    # for the actual launch command and CYCLE_PLAN_TTS.md "Risk Mitigation" section.
    K_train: int = 2  # inner steps during meta-training
    K_test: int = 3  # inner steps at test time
    nfe_forward_inner: int = 5  # NFE for forward gen inside inner loop (was 6)
    nfe_cycle_inner: int = 3   # NFE for cycle gen inside inner loop (was 4)
    nfe_outer: int = 10        # NFE for outer-loop *meta-gradient* gen (was 16)
    nfe_test_final: int = 32   # NFE for final eval-time generation (paper quality)
    cfg_strength_inner: float = 1.0  # CFG=1 disables it; cheaper inside inner loop
    cfg_strength_outer: float = 2.0
    truncate_cycle_grad: bool = True  # detach ŷ on the boundary
    residual_grad_lr: float = 5e-4  # first-order fallback so ψ cannot learn a no-op
    use_outer_grad_clip: float = 1.0
    inner_loss_lambda: dict = field(
        default_factory=lambda: {
            "spk": 1.0,    # L_spk (cycle speaker cosine)
            "spec": 1.0,   # L_spec (cycle mel L1)
            "f0": 1.0,     # L_f0  (cycle F0 Pearson)
            "id": 1.0,     # L_id  (forward speaker cosine)
            "intel": 0.0,  # L_intel — not used in inner loop (non-diff CER); see plan §6
        }
    )
    outer_lambda_cer: float = 0.5
    outer_lambda_f0: float = 0.3


@dataclass
class StabilityConfig:
    """Bounds that keep LoRA from dominating the frozen F5 backbone."""

    max_A_abs: float = 0.08          # per-element cap on A (pristine σ≈0.02)
    max_B_abs: float = 0.05          # keep B small — identity start has B=0
    max_AB_product_norm: float = 0.15  # cap ‖A‖_F · ‖B‖_F per layer
    max_psi_update_norm: float = 0.02  # global norm clip on each ψ Δθ tensor
    init_anchor_strength: float = 0.05  # mix θ₀ toward pristine after Reptile
    max_adam_step_norm: float = 0.02   # test-time Adam (B2) step cap


# EMNLP v3: slightly more adapter capacity + stronger L_id (still stable).
STABILITY_EMNLP_V3 = StabilityConfig(
    max_A_abs=0.10,
    max_B_abs=0.12,
    max_AB_product_norm=0.22,
    max_psi_update_norm=0.02,
    init_anchor_strength=0.03,
    max_adam_step_norm=0.02,
)

INNER_LOSS_EMNLP_V3 = {
    "spk": 1.0,
    "spec": 0.5,
    "f0": 0.5,
    "id": 2.0,   # emphasize forward identity (L_id)
    "intel": 0.0,
}


@dataclass
class TrainingConfig:
    # 2-day A100 budget targets ~ M=900 meta-iters with B=4 episodes/iter.
    # Each iter ≈ 90-100 s of GPU at NFE_fwd=5 / NFE_cyc=3 / NFE_outer=10 / K=2.
    M: int = 900            # total meta-iterations (was 1500)
    B: int = 4              # episodes per meta-batch
    lr_optim: float = 1e-4  # for ψ and φ (Adam) — reduced for stability
    lr_init: float = 3e-4   # for θ₀ (Reptile step size, applied as Reptile β)
    reptile_beta: float = 0.1  # small outer meta-step on θ₀ (was 0.3)
    weight_decay: float = 0.0
    warmup_steps: int = 50
    grad_clip: float = 0.5
    skip_nan_iters: bool = True
    log_every: int = 5
    ckpt_every: int = 100
    eval_every: int = 200   # quick val (~2-3 min each) every 200 iters
    seed: int = 1337


@dataclass
class DataConfig:
    sample_rate: int = 24000
    prompt_min_sec: float = 3.0
    prompt_max_sec: float = 7.0
    target_min_sec: float = 4.0
    target_max_sec: float = 8.0
    cache_dir: Path = ROOT / "data" / "cache" / "feats"
    manifest_dir: Path = ROOT / "data" / "manifests"
    seed: int = 1337


@dataclass
class CycleAdaptConfig:
    """Top-level config bundle."""

    iaa: IAAConfig = field(default_factory=IAAConfig)
    psi: MetaOptConfig = field(default_factory=MetaOptConfig)
    phi: LossWeighterConfig = field(default_factory=LossWeighterConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    stab: StabilityConfig = field(default_factory=StabilityConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Languages used (must align with phi.n_languages indices).
    # We TRAIN only on en/zh (rich per-speaker corpora exist), and EVAL
    # zero-shot on {es, fr, de, hi, ja} + the seen langs.  This frames the
    # paper as "trained on 2 languages, generalises to 7" — a much stronger
    # zero-shot story than "trained on 3 langs, tested on 1 zero-shot lang".
    languages: tuple = ("en", "zh", "es", "fr", "de", "hi", "ja")
    lang_idx: dict = field(
        default_factory=lambda: {"en": 0, "zh": 1, "es": 2, "fr": 3, "de": 4, "hi": 5, "ja": 6, "ko": 7}
    )
    # Meta-training language pairs (sampled with these weights).
    # Only L1∈{en,zh} because those are the langs with multi-utterance speaker
    # data.  L2∈{en,zh} keeps the cross-lingual signal but doesn't bias the
    # model toward any specific eval language.
    train_lang_pairs: tuple = (
        ("en", "zh", 1.0),
        ("zh", "en", 1.0),
        ("en", "en", 0.3),  # mono-lingual sanity ratio
        ("zh", "zh", 0.3),
    )
    # Eval language pairs evaluated at the end of training.
    eval_lang_pairs: tuple = (
        ("en", "en", "in-distrib"),
        ("en", "zh", "in-distrib"),
        ("zh", "en", "in-distrib"),
        ("zh", "zh", "in-distrib"),
        ("en", "es", "zero-shot"),
        ("en", "fr", "zero-shot"),
        ("en", "de", "zero-shot"),
        ("en", "hi", "zero-shot"),
        ("en", "ja", "zero-shot"),
        ("zh", "ja", "zero-shot"),
    )

    # Logging / paths
    run_name: str = "cycleadapt_v1"
    ckpt_dir: Path = ROOT / "checkpoints"
    log_dir: Path = ROOT / "logs"
    results_dir: Path = ROOT / "results"

    def apply_emnlp_v3_preset(self) -> None:
        """Stronger L_id + relaxed (but bounded) LoRA for Findings experiments."""
        from dataclasses import replace
        self.stab = replace(STABILITY_EMNLP_V3)
        self.cycle.inner_loss_lambda = dict(INNER_LOSS_EMNLP_V3)
        self.train.reptile_beta = 0.12
        self.psi.output_scale = 1e-4

    def __post_init__(self) -> None:
        self.ckpt_dir = Path(self.ckpt_dir) / self.run_name
        self.log_dir = Path(self.log_dir) / self.run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.run_name.endswith("emnlp_v3") or "cycleadapt_v3" in self.run_name:
            self.apply_emnlp_v3_preset()
        # v3: allow a slightly stronger adapter + emphasize L_id (EMNLP re-run).
        if "v3" in self.run_name:
            self.stab.max_B_abs = 0.12
            self.stab.max_AB_product_norm = 0.28
            self.stab.init_anchor_strength = 0.03
            self.stab.max_psi_update_norm = 0.03
            self.train.reptile_beta = 0.15
            self.cycle.inner_loss_lambda = {
                "spk": 0.8, "spec": 0.5, "f0": 0.5, "id": 2.0, "intel": 0.0,
            }
