# CycleAdapt-TTS: Meta-Learned Cycle-Consistent Test-Time Training for Cross-Lingual Speaker Identity Preservation

## Research Plan — EMNLP 2026 Submission

---

## 1. Problem Statement

Cross-lingual zero-shot text-to-speech (TTS) systems clone a speaker's voice from a short audio prompt and synthesize speech in a different language. Despite rapid progress in multilingual TTS (XTTS, VALL-E X, VoiceCraft-X, F5-TTS), **speaker identity degrades when the target language differs from the prompt language** — the output sounds less like the original speaker, exhibiting timbre drift, prosodic contamination, and spectral averaging. Current systems treat this as an unavoidable artifact of cross-lingual transfer and offer no mechanism for inference-time correction.

We propose **CycleAdapt-TTS**, a meta-learned test-time training framework that self-corrects speaker identity drift at inference using cycle-consistent self-supervised losses. Given a speaker's prompt in language $L_1$ and target text in language $L_2$, the system generates $L_2$ speech, re-synthesizes it back into $L_1$, and compares the round-tripped audio with the original prompt. A meta-learned optimizer uses this cycle-consistency signal to update a lightweight identity alignment adapter in 1–3 gradient steps, producing identity-corrected output — all without labeled data, speaker-specific training, or access to bilingual recordings.

---

## 2. Core Contributions

1. **Identity Alignment Adapter (IAA):** A lightweight LoRA-style module injected into the speaker conditioning pathway of a frozen multilingual TTS, meta-trained episodically for cross-lingual identity transfer.

2. **Cycle-Consistent Self-Supervised Test-Time Training:** A round-trip generation loop ($L_1 \to L_2 \to L_1$) that provides a rich, label-free error signal for identity preservation — the first application of cycle consistency to TTS.

3. **Meta-Learned Adaptation Dynamics:** Both the optimizer (update rule) and the loss weighting function are meta-learned, enabling stable convergence in 1–3 steps for any unseen speaker.

4. **Novel Evaluation Metrics:** Cycle Consistency Gap and Compounding Ratio — two new metrics that quantify identity degradation through bidirectional cross-lingual transfer.

---

## 3. Datasets

### 3.1 Training Datasets (for Meta-Training Episodes)

#### 3.1.1 Multilingual LibriSpeech (MLS)

- **Languages:** English (EN), Spanish (ES), French (FR), German (DE), Dutch, Italian, Portuguese, Polish
- **Size:** ~50,000 hours total; EN subset ~44,000 hours, ES subset ~918 hours
- **Speakers:** ~6,000+ across all languages
- **Use:** Primary source for EN and ES speakers in meta-training episodes
- **Format:** 16kHz FLAC, sentence-level alignments provided
- **Download:**
```bash
# Full MLS (large — download only needed languages)
wget https://dl.fbaipublicfiles.com/mls/mls_english.tar.gz        # ~900GB (use opus subset below instead)
wget https://dl.fbaipublicfiles.com/mls/mls_spanish.tar.gz

# Recommended: Use the smaller MLS subsets or OpenSLR mirrors
# English 10hr subset for prototyping:
wget https://dl.fbaipublicfiles.com/mls/mls_english_opus.tar.gz   # ~120GB opus-compressed

# Or via HuggingFace (streaming, no full download needed):
# Dataset ID: facebook/multilingual_librispeech
```
- **Preprocessing:** Resample to 24kHz (F5-TTS requirement), segment into 3–30s utterances, extract speaker IDs.

#### 3.1.2 AISHELL-3

- **Languages:** Mandarin Chinese (ZH)
- **Size:** ~85 hours
- **Speakers:** 218 speakers (176 female, 42 male)
- **Use:** Chinese speakers for EN↔ZH and ES↔ZH meta-training episodes
- **Format:** 44.1kHz WAV
- **Download:**
```bash
# From OpenSLR
wget https://www.openslr.org/resources/93/data_aishell3.tgz
# Approx 18GB compressed

# Or via HuggingFace:
# Dataset ID: aishell3/aishell3
```
- **Preprocessing:** Downsample to 24kHz, filter utterances < 2s or > 30s, verify speaker labels.

#### 3.1.3 IndicTTS Database (IITM)

- **Languages:** Hindi (HI) + 12 other Indian languages
- **Size:** ~10–40 hours for Hindi
- **Speakers:** 1 male + 1 female per language (limited speaker diversity)
- **Use:** Hindi speakers for EN↔HI episodes
- **Format:** 48kHz WAV
- **Download:**
```bash
# From IIT Madras:
# https://www.iitm.ac.in/donlab/tts/database.php
# Requires registration — fill the form, download link emailed

# Alternative: IndicVoices (larger, more speakers):
# https://huggingface.co/datasets/ai4bharat/IndicVoices
# ~7,000 hours across 22 languages, open access
```
- **Preprocessing:** Resample to 24kHz, select Hindi subset, segment.
- **Note:** If IndicTTS speaker diversity is too low (only 2 speakers), supplement with IndicVoices or LIMMITS dataset from IIT Madras which has more Hindi speakers.

#### 3.1.4 LibriTTS-R

- **Languages:** English (EN)
- **Size:** ~585 hours (train-clean-100 + train-clean-360 subsets recommended)
- **Speakers:** ~2,400 speakers
- **Use:** High-quality English multi-speaker data, supplements MLS for speaker diversity
- **Format:** 24kHz WAV (already at target sample rate)
- **Download:**
```bash
# LibriTTS-R (restored, higher quality than original LibriTTS)
wget https://www.openslr.org/resources/141/train-clean-100.tar.gz   # ~7.5GB
wget https://www.openslr.org/resources/141/train-clean-360.tar.gz   # ~26GB

# Or via HuggingFace:
# Dataset ID: cdminix/libritts-r-aligned
```

### 3.2 Evaluation Datasets (Held-Out, Never Seen in Training)

#### 3.2.1 VCTK

- **Languages:** English (EN)
- **Size:** ~44 hours
- **Speakers:** 110 speakers with various accents
- **Use:** Primary held-out speaker set for evaluation
- **Download:**
```bash
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
# ~11GB

# Or via HuggingFace:
# Dataset ID: CSTR-Edinburgh/vctk
```

#### 3.2.2 Common Voice (v16+) — ES, ZH, HI Subsets

- **Languages:** Spanish, Chinese (zh-CN), Hindi
- **Size:** Variable per language (ES ~400hrs validated, ZH ~200hrs, HI ~20hrs)
- **Speakers:** Thousands per language
- **Use:** Held-out speakers for cross-lingual evaluation in target languages
- **Download:**
```bash
# Requires Mozilla Common Voice account (free)
# https://commonvoice.mozilla.org/en/datasets
# Download validated subsets for: es, zh-CN, hi

# Or via HuggingFace:
# Dataset ID: mozilla-foundation/common_voice_16_1
# Select language config: es, zh-CN, hi
```
- **Preprocessing:** Select 50 speakers per language with > 30 utterances each, hold out completely.

#### 3.2.3 FLEURS (for Zero-Shot Language Pair Evaluation)

- **Languages:** 102 languages
- **Size:** ~12 hours per language
- **Use:** Additional language pairs for zero-shot generalization testing (e.g., ES→ZH pair not seen in meta-training)
- **Download:**
```bash
# Via HuggingFace:
# Dataset ID: google/fleurs
# Select: es_419, cmn_hans_cn
```

### 3.3 Data for Domain Shift Experiments

No additional dataset needed. Apply augmentations to evaluation prompts using:

```bash
pip install audiomentations sox

# Augmentations applied at test time to evaluation prompts:
# 1. Additive noise: SNR = {5, 10, 20} dB (white noise, babble noise from MUSAN)
# 2. Reverberation: RT60 = {0.3, 0.6, 1.0} seconds (room impulse responses from RIR_NOISES)
# 3. Telephone band filtering: 300–3400 Hz bandpass
# 4. Codec compression: MP3 at 32kbps

# Download MUSAN for noise:
wget https://www.openslr.org/resources/17/musan.tar.gz  # ~11GB

# Download RIR_NOISES for reverberation:
wget https://www.openslr.org/resources/28/rirs_noises.zip  # ~6GB
```

### 3.4 Dataset Summary Table

| Dataset | Lang | Hours | Speakers | Split | Purpose |
|---------|------|-------|----------|-------|---------|
| MLS (EN + ES) | EN, ES | ~45,000 | ~5,000 | Train | Meta-training episodes |
| AISHELL-3 | ZH | 85 | 218 | Train | Meta-training episodes |
| IndicVoices (HI) | HI | ~200 | ~1,000 | Train | Meta-training episodes |
| LibriTTS-R | EN | 460 | 2,400 | Train | Supplement speaker diversity |
| VCTK | EN | 44 | 110 | **Eval** | Held-out English speakers |
| Common Voice (ES, ZH, HI) | ES, ZH, HI | Variable | 50/lang | **Eval** | Held-out cross-lingual speakers |
| FLEURS | ES, ZH | ~12/lang | Variable | **Eval** | Zero-shot pair evaluation |

---

## 4. Base Model and Architecture

### 4.1 Frozen Backbone: F5-TTS

F5-TTS is a non-autoregressive flow-matching TTS that performs text-guided speech infilling. It uses a Diffusion Transformer (DiT) with ConvNeXt-based text modeling and supports zero-shot voice cloning via audio prompting.

- **Architecture:** DiT with adaLN-zero, ConvNeXt V2 text encoder
- **Parameters:** ~330M (all frozen)
- **Input:** Text (character-level) + audio prompt (mel spectrogram)
- **Output:** Mel spectrogram → vocoder (Vocos) → waveform
- **Supported languages:** EN, ZH natively; extensible to others via Cross-Lingual F5-TTS
- **Repository:** `https://github.com/SWivid/F5-TTS`

### 4.2 Identity Alignment Adapter (IAA)

A LoRA-style low-rank adaptation module injected into the speaker conditioning pathway of F5-TTS.

**Injection points:** The DiT blocks where the audio prompt's speaker information conditions the generation. Specifically, the cross-attention layers and the adaLN modulation layers that incorporate speaker context.

**Architecture:** For each target weight matrix $W \in \mathbb{R}^{d \times d}$ in the speaker conditioning path:

$$W' = W + \Delta W = W + B A$$

where $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, and $r \ll d$ is the adapter rank.

**Hyperparameters:**
- Rank $r = 8$ (ablate over $\{4, 8, 16, 32\}$)
- Number of adapted layers: top 4–6 DiT blocks (ablate)
- Total adapter parameters: ~0.5–1.5M depending on rank and number of layers

**Initialization:** $A$ initialized from $\mathcal{N}(0, \sigma^2)$, $B$ initialized to zero, so $\Delta W = 0$ at initialization (adapter starts as identity). Meta-training learns a better $\theta_0 = \{A_0, B_0\}$ that serves as a universal starting point.

### 4.3 Learned Optimizer $\psi$

A small MLP that maps gradient information to parameter updates, replacing standard SGD/Adam.

**Input to $\psi$:**
- Gradient $g = \nabla_\theta \mathcal{L}_{\text{total}}$ (per-parameter)
- Current total loss $\mathcal{L}_{\text{total}}$
- Step index $k$ (positionally encoded)

**Architecture:**

$$\Delta\theta = \psi(g, \mathcal{L}_{\text{total}}, k) = \text{MLP}\left([\log(|g| + \epsilon),\; \text{sign}(g),\; \mathcal{L}_{\text{total}},\; \text{PE}(k)]\right)$$

Following the learned optimizer literature (Andrychowicz et al., 2016), we operate on $\log|g|$ and $\text{sign}(g)$ rather than raw gradients for numerical stability.

**MLP structure:** 2 hidden layers, 64 units each, ReLU activation, output dimension matches input gradient dimension. Coordinate-wise: the same MLP is applied independently to each parameter coordinate (weight sharing across coordinates).

**Parameters:** ~50–100K total.

### 4.4 Cycle Loss Weighting Network $\phi$

A small network that takes the vector of individual sub-losses and the language pair encoding, and outputs per-loss weights.

**Input:**
- Loss vector $\ell = [\mathcal{L}_{\text{spk}},\; \mathcal{L}_{\text{spec}},\; \mathcal{L}_{f_0},\; \mathcal{L}_{\text{id}},\; \mathcal{L}_{\text{intel}}]$
- Language pair embedding $e_{L_1, L_2}$ (learned embeddings for each language, concatenated)

**Architecture:**

$$w = \text{softmax}\left(\text{MLP}\left([\ell;\; e_{L_1};\; e_{L_2}]\right)\right)$$

**MLP structure:** 2 hidden layers, 32 units each, output dimension 5 (one weight per sub-loss). Softmax ensures weights sum to 1.

**Parameters:** ~30–50K total.

### 4.5 Pre-Trained Feature Extractors (All Frozen, Off-the-Shelf)

| Extractor | Model | Purpose | Source |
|-----------|-------|---------|--------|
| Speaker encoder | WavLM-TDNN (SpeechBrain) | Compute SIM-o / SECS | `speechbrain/spkrec-ecapa-voxceleb` |
| Speaker encoder 2 | Resemblyzer (GE2E) | Compute SECS (secondary) | `resemblyzer` PyPI package |
| ASR | Whisper-large-v3 | Compute WER/CER, transcribe prompts | `openai/whisper-large-v3` |
| F0 extractor | CREPE | Extract fundamental frequency | `crepe` PyPI package |
| MOS predictor | UTMOS | Predict speech quality | `sarulab-speech/UTMOS-demo` |

---

## 5. Mathematical Formulation

### 5.1 Notation

| Symbol | Meaning |
|--------|---------|
| $G(\cdot;\theta)$ | Frozen TTS backbone with adapter parameters $\theta$ |
| $x$ | Speaker prompt audio in language $L_1$ |
| $t$ | Target text in language $L_2$ |
| $\hat{y}$ | Generated speech in $L_2$ |
| $t'$ | Source language text (transcript of $x$, obtained via Whisper) |
| $\hat{y}'$ | Cycle-reconstructed speech in $L_1$ (from $\hat{y}$ as prompt) |
| $\text{SpkEnc}(\cdot)$ | Speaker encoder (WavLM-TDNN) |
| $F_0(\cdot)$ | Fundamental frequency extractor |
| $\text{ASR}(\cdot)$ | Whisper transcription |
| $\theta_0$ | Meta-learned adapter initialization |
| $\psi$ | Meta-learned optimizer |
| $\phi$ | Meta-learned loss weighting network |
| $K$ | Number of inner-loop (test-time) adaptation steps |

### 5.2 Cycle-Consistent Self-Supervised Losses

Given prompt $x$ in $L_1$, we generate $\hat{y}$ in $L_2$, then cycle back to produce $\hat{y}'$ in $L_1$:

$$\hat{y} = G(x, t, L_2;\; \theta)$$

$$\hat{y}' = G(\hat{y}, t', L_1;\; \theta)$$

We define five self-supervised sub-losses, grouped into **cycle losses** (comparing $x$ with $\hat{y}'$) and **forward losses** (comparing $x$ with $\hat{y}$):

**Cycle Losses (round-trip signal):**

$$\mathcal{L}_{\text{spk}} = 1 - \frac{\text{SpkEnc}(x) \cdot \text{SpkEnc}(\hat{y}')}{\|\text{SpkEnc}(x)\| \cdot \|\text{SpkEnc}(\hat{y}')\|}$$

$$\mathcal{L}_{\text{spec}} = \frac{1}{T} \sum_{t=1}^{T} \| \text{Mel}(x)_t - \text{Mel}(\hat{y}')_t \|_1$$

$$\mathcal{L}_{f_0} = 1 - \rho\left(F_0(x),\; F_0(\hat{y}')\right)$$

where $\rho(\cdot, \cdot)$ denotes Pearson correlation computed over voiced frames after DTW alignment.

**Forward Losses (direct signal):**

$$\mathcal{L}_{\text{id}} = 1 - \frac{\text{SpkEnc}(x) \cdot \text{SpkEnc}(\hat{y})}{\|\text{SpkEnc}(x)\| \cdot \|\text{SpkEnc}(\hat{y})\|}$$

$$\mathcal{L}_{\text{intel}} = \text{CER}\left(\text{ASR}(\hat{y}),\; t\right)$$

### 5.3 Meta-Learned Loss Aggregation

The total loss at each inner-loop step is a weighted combination:

$$\mathcal{L}_{\text{total}} = \sum_{i=1}^{5} w_i \cdot \mathcal{L}_i$$

where the weights are produced by the meta-learned weighting network:

$$\mathbf{w} = \phi\left([\mathcal{L}_1, \ldots, \mathcal{L}_5];\; e_{L_1};\; e_{L_2}\right) \in \Delta^4$$

and $\Delta^4$ is the 4-simplex (weights are non-negative and sum to 1).

### 5.4 Meta-Learned Optimizer Update

At each inner-loop step $k$, the adapter is updated:

$$g_k = \nabla_\theta \mathcal{L}_{\text{total}}^{(k)}$$

$$\theta_{k+1} = \theta_k - \psi\left(\log|g_k| + \epsilon,\; \text{sign}(g_k),\; \mathcal{L}_{\text{total}}^{(k)},\; k\right)$$

where $\psi$ is applied coordinate-wise with shared parameters.

### 5.5 Meta-Training Objective

Each meta-training episode samples a task $\tau = (s, L_1, L_2)$ consisting of a speaker $s$ and a language pair. The episode contains:
- **Support data:** prompt $x^{(\tau)}$ in $L_1$, target text $t^{(\tau)}$ in $L_2$
- **Query data:** different target text $t_q^{(\tau)}$ in $L_2$ (for outer-loop evaluation)

The inner loop simulates test-time adaptation:

$$\theta_K^{(\tau)} = \text{InnerLoop}(\theta_0, x^{(\tau)}, t^{(\tau)}, L_1, L_2;\; \psi, \phi, K)$$

The outer-loop loss evaluates the adapted adapter on the query:

$$\hat{y}_q = G(x^{(\tau)}, t_q^{(\tau)}, L_2;\; \theta_K^{(\tau)})$$

$$\mathcal{L}_{\text{outer}}^{(\tau)} = \underbrace{(1 - \text{SIM-o}(x^{(\tau)}, \hat{y}_q))}_{\text{identity}} + \lambda_1 \underbrace{\text{CER}(\text{ASR}(\hat{y}_q), t_q^{(\tau)})}_{\text{intelligibility}} + \lambda_2 \underbrace{(1 - \rho(F_0(x^{(\tau)}), F_0(\hat{y}_q)))}_{\text{prosody}}$$

The meta-parameters are updated by aggregating over a batch of $B$ episodes:

$$\mathcal{L}_{\text{meta}} = \frac{1}{B} \sum_{\tau=1}^{B} \mathcal{L}_{\text{outer}}^{(\tau)}$$

$$\theta_0 \leftarrow \theta_0 - \alpha \nabla_{\theta_0} \mathcal{L}_{\text{meta}}$$

$$\psi \leftarrow \psi - \alpha \nabla_\psi \mathcal{L}_{\text{meta}}$$

$$\phi \leftarrow \phi - \alpha \nabla_\phi \mathcal{L}_{\text{meta}}$$

### 5.6 First-Order Approximation (Reptile Variant)

Full MAML requires second-order gradients through the inner loop, which is memory-intensive. We use a first-order approximation:

$$\theta_0 \leftarrow \theta_0 + \beta \cdot \frac{1}{B} \sum_{\tau=1}^{B} (\theta_K^{(\tau)} - \theta_0)$$

For $\psi$ and $\phi$, we use first-order MAML (stop-gradient through the inner loop's dependency on $\psi, \phi$, but still compute outer-loop gradients). This is standard practice in meta-learning when memory is constrained.

### 5.7 Full Algorithms

#### Algorithm 1: Meta-Training

```
Input: Frozen TTS G, datasets D, inner steps K, meta-learning rate α, 
       episode batch size B, outer loss weights λ₁, λ₂
Initialize: θ₀ (adapter), ψ (optimizer), φ (loss weighter)

For each meta-iteration m = 1, 2, ..., M:

    For each episode τ = 1, ..., B:

        Sample speaker s, languages L₁, L₂ from D
        Sample prompt x in L₁ spoken by s
        Sample support text t and query text t_q in L₂
        Obtain t' = ASR(x)  // Whisper transcript of prompt
        
        // --- Inner Loop (simulated test-time training) ---
        θ ← θ₀
        
        For k = 1, ..., K:
            ŷ     = G(x, t, L₂; θ)           // forward generation
            ŷ'    = G(ŷ, t', L₁; θ)           // cycle reconstruction
            
            Compute L_spk, L_spec, L_f0, L_id, L_intel
            w     = φ([L_spk, L_spec, L_f0, L_id, L_intel], e_L1, e_L2)
            L_tot = Σᵢ wᵢ · Lᵢ
            
            g     = ∇_θ L_tot
            θ     ← θ - ψ(log|g|, sign(g), L_tot, k)
        End inner loop
        
        // --- Outer Loop Evaluation ---
        ŷ_q = G(x, t_q, L₂; θ)
        
        L_outer(τ) = (1 - SIM-o(x, ŷ_q)) 
                    + λ₁ · CER(ASR(ŷ_q), t_q) 
                    + λ₂ · (1 - F0_PCC(x, ŷ_q))
    
    End episode loop
    
    // --- Meta-Update ---
    L_meta = (1/B) Σ_τ L_outer(τ)
    
    // First-order updates:
    θ₀ ← θ₀ + β · (1/B) Σ_τ (θ_K(τ) - θ₀)      // Reptile for adapter
    ψ  ← ψ  - α · ∇_ψ  L_meta                     // first-order MAML for optimizer
    φ  ← φ  - α · ∇_φ  L_meta                     // first-order MAML for weighter

End meta-iteration

Output: θ₀*, ψ*, φ*
```

#### Algorithm 2: Test-Time Inference

```
Input: Unseen speaker prompt x in L₁, target text t in L₂,
       meta-learned θ₀*, ψ*, φ*, frozen TTS G, inner steps K

// Obtain source language transcript
t' = ASR(x)                               // Whisper transcription

// Initialize adapter
θ ← θ₀*

// --- Test-Time Training Loop ---
For k = 1, ..., K:
    ŷ     = G(x, t, L₂; θ)                // forward generation
    ŷ'    = G(ŷ, t', L₁; θ)               // cycle reconstruction
    
    Compute L_spk, L_spec, L_f0, L_id, L_intel
    w     = φ*([L_spk, L_spec, L_f0, L_id, L_intel], e_L1, e_L2)
    L_tot = Σᵢ wᵢ · Lᵢ
    
    g     = ∇_θ L_tot
    θ     ← θ - ψ*(log|g|, sign(g), L_tot, k)
End loop

// --- Final Generation ---
ŷ_final = G(x, t, L₂; θ)

Output: ŷ_final
```

---

## 6. Baselines

### 6.1 External Baselines

| ID | System | Description | What It Tests |
|----|--------|-------------|---------------|
| B1 | **F5-TTS (vanilla)** | Frozen F5-TTS, zero-shot prompting, no adaptation | Lower bound — raw cross-lingual capability |
| B2 | **Cross-Lingual F5-TTS** | F5-TTS + speaking-rate predictors from the 2025 paper | State-of-the-art non-adaptive cross-lingual baseline |
| B3 | **LoRP-TTS** | LoRA finetuning at inference time, 100 naive SGD steps, no meta-learning | Tests whether meta-learning matters vs. brute-force test-time finetuning |
| B4 | **XTTS v2** | Coqui XTTS, zero-shot prompting, 16 languages | Industry-grade multilingual baseline |

### 6.2 Ablation Systems (Your Own Variants)

| ID | System | What's Removed | What It Tests |
|----|--------|----------------|---------------|
| A1 | **IAA only** | No TTA, no cycle; adapter trained episodically, used as-is at test time | Does the adapter alone help? |
| A2 | **IAA + naive GD** | No meta-learned optimizer; use Adam with lr=1e-4 for K=3 steps at test time using cycle loss | Does meta-learning the optimizer matter? |
| A3 | **IAA + meta-optimizer, no cycle** | No cycle loss; only forward $\mathcal{L}_{\text{id}}$ + $\mathcal{L}_{\text{intel}}$ for TTA | Does cycle consistency add value beyond forward losses? |
| A4 | **IAA + meta-optimizer + cycle, uniform weights** | No learned $\phi$; all 5 losses weighted equally | Does meta-learning the loss weights matter? |
| A5 | **Full CycleAdapt-TTS** | Nothing removed | Full system |

---

## 7. Experiments

### 7.1 Language Pairs

| Pair | Training Status | Linguistic Motivation |
|------|----------------|----------------------|
| EN → ZH | Seen in meta-training | Non-tonal → tonal; maximal phonological distance |
| EN → ES | Seen in meta-training | Stress-timed → syllable-timed; same script family |
| EN → HI | Seen in meta-training | Indo-European but distant phonotactics; Indian language TTS comparison (VECL-TTS setting) |
| ZH → EN | Seen in meta-training | Reverse of EN→ZH; tests directional asymmetry of cycle consistency |
| ES → ZH | **Held out** | Never seen in meta-training; tests zero-shot language pair generalization |

### 7.2 Experiment 1: Main Cross-Lingual Identity Preservation

**Setup:** For each of 5 language pairs, evaluate all 9 systems (B1–B4, A1–A5) on held-out speakers from VCTK (EN prompts) and Common Voice (ES, ZH, HI prompts). Generate 200 utterances per language pair per system (50 speakers × 4 utterances each).

**Metrics reported per pair:** SIM-o, SECS, WER, CER, F0 PCC, UTMOS, Cycle Gap, Compounding Ratio.

**Expected main result:** A5 (full system) > A3 (no cycle) > A2 (naive GD) > A1 (no TTA) > B1 (vanilla), demonstrating incremental value of each component. A5 should also beat B2, B3, B4 on identity metrics while maintaining comparable or better intelligibility.

### 7.3 Experiment 2: Ablation of Cycle Consistency Components

**Setup:** Decompose the cycle loss into its three sub-components and test variants with different subsets:

| Variant | $\mathcal{L}_{\text{spk}}$ | $\mathcal{L}_{\text{spec}}$ | $\mathcal{L}_{f_0}$ | Forward losses |
|---------|:---:|:---:|:---:|:---:|
| Forward only | ✗ | ✗ | ✗ | ✓ |
| Cycle-spk only | ✓ | ✗ | ✗ | ✓ |
| Cycle-spk+spec | ✓ | ✓ | ✗ | ✓ |
| Cycle-spk+F0 | ✓ | ✗ | ✓ | ✓ |
| Full cycle | ✓ | ✓ | ✓ | ✓ |

**Expected result:** Full cycle > any subset. $\mathcal{L}_{\text{spk}}$ provides the largest individual gain; $\mathcal{L}_{f_0}$ is critical for tonal languages (EN→ZH) but less so for EN→ES; $\mathcal{L}_{\text{spec}}$ provides consistent but smaller improvements.

### 7.4 Experiment 3: Adaptation Convergence

**Setup:** For A2 (naive GD), A3 (meta-optimizer without cycle), and A5 (full system), plot SIM-o and WER as a function of TTA steps $K \in \{0, 1, 2, 3, 5, 10, 20, 50\}$.

**Expected result:**
- A5 converges to near-optimal SIM-o within $K=2$–$3$ steps.
- A2 (naive GD) requires $K=50$–$100$ steps to reach similar SIM-o (matching LoRP-TTS findings of ~100 steps).
- A3 converges faster than A2 but slower than A5, showing cycle loss provides a better optimization landscape.
- All systems show WER degradation at very high $K$ (overfitting), but A5 degrades less due to meta-learned stability.

**Figure:** Line plot with $K$ on x-axis, SIM-o on left y-axis, WER on right y-axis, three curves (A2, A3, A5).

### 7.5 Experiment 4: Zero-Shot Language Pair Generalization

**Setup:** Evaluate on ES→ZH (never seen in meta-training). Compare B1, B2, A1, and A5.

**Expected result:** A5 still improves identity over B1 and A1, even though the language pair was never seen. This demonstrates that the meta-learned adaptation dynamics (optimizer ψ, weighting φ) are language-pair agnostic — they learn how to fix identity drift in general, not for specific language pairs. The improvement may be smaller than for seen pairs but should still be statistically significant.

### 7.6 Experiment 5: Prompt Duration Sensitivity

**Setup:** Vary prompt audio duration: 3s, 6s, 10s, 30s. Evaluate A5 vs. B1 vs. B2 on EN→ZH.

**Expected result:** At 3s (very short prompt), identity is weakest for all systems. A5 shows the largest relative improvement over baselines at short prompt durations, because the cycle consistency loop can extract more identity information from the generated audio than a single short prompt provides. At 30s, B1 and B2 are already quite good, and A5's improvement is smaller (diminishing returns).

**Figure:** Bar plot grouped by prompt duration, bars for each system, SIM-o on y-axis.

### 7.7 Experiment 6: Domain Shift Robustness

**Setup:** Apply noise (SNR 5/10/20 dB), reverberation (RT60 0.3/0.6/1.0s), and telephone-band filtering to evaluation prompts. Evaluate A5 vs. B1 vs. B3 on EN→ZH.

**Expected result:** All systems degrade under noise, but A5 degrades less because the TTA loop can partially correct for the noisy prompt. The cycle consistency signal is more robust to noise than forward-only SECS, because noise artifacts in the prompt propagate and amplify through the cycle, making them easier to detect and correct.

**Figure:** Line plot with noise SNR on x-axis, SIM-o on y-axis, curves for A5 vs. baselines.

### 7.8 Experiment 7: Learned Loss Weight Analysis

**Setup:** After meta-training, visualize the weights $\mathbf{w}$ output by $\phi$ across different language pairs and noise conditions. Run $\phi$ on 500 evaluation episodes and aggregate.

**Expected results:**
- For EN→ZH (tonal target), $\phi$ upweights $\mathcal{L}_{f_0}$ because tonal languages leak F0 patterns into identity.
- For EN→ES (same script family), $\phi$ downweights $\mathcal{L}_{f_0}$ and upweights $\mathcal{L}_{\text{spec}}$.
- Under noisy prompts, $\phi$ downweights $\mathcal{L}_{\text{spec}}$ (noisy spectral comparisons are unreliable) and upweights $\mathcal{L}_{\text{spk}}$ (speaker embeddings are more noise-robust).

**Figure:** Heatmap — rows are language pairs / conditions, columns are loss components, cell values are average weights. This is an interpretability result.

### 7.9 Experiment 8: Computational Overhead Analysis


**Report:**
| System | GPU | Time per utterance | Overhead vs. vanilla |
|--------|-----|-------------------|---------------------|
| B1 (vanilla F5-TTS) | A100 | ~2s | 1.0× |
| A5 (K=1) | A100 | ~5s | ~2.5× |
| A5 (K=3) | A100 | ~10s | ~5× |
| B3 (LoRP, 100 steps) | A100 | ~90s | ~45× |
| A5 (K=3) | T4 | ~25s | — |

**Key argument:** 5× overhead is comparable to classifier-free guidance in diffusion models (which doubles inference cost and is universally accepted). 45× for LoRP-TTS is impractical; our meta-learned approach achieves comparable or better results in 1/30th the time.

---

## 8. Evaluation Metrics — Full Specification

### 8.1 Identity Fidelity

**SIM-o (primary):** Speaker embedding cosine similarity using WavLM-TDNN from SpeechBrain. This is the 2024–2025 gold standard used by VoiceCraft-X and Cross-Lingual F5-TTS.

$$\text{SIM-o}(x, \hat{y}) = \frac{\text{WavLM-TDNN}(x) \cdot \text{WavLM-TDNN}(\hat{y})}{\|\text{WavLM-TDNN}(x)\| \cdot \|\text{WavLM-TDNN}(\hat{y})\|}$$

**SECS (secondary):** Speaker embedding cosine similarity using Resemblyzer (GE2E model). Reported for comparability with XTTS and MultiVerse.

**EER (Speaker Verification Equal Error Rate):** Treat (prompt, generated) pairs as same-speaker trials and (prompt, different-speaker-generated) as different-speaker trials. Lower EER = better identity preservation. Follows Meta-TTS evaluation protocol.

### 8.2 Intelligibility

**WER / CER:** Transcribe generated audio with Whisper-large-v3, compute word/character error rate against the target text $t$. Report per-language.

### 8.3 Prosody Preservation

**F0 PCC:** Pearson correlation coefficient between F0 contours of prompt $x$ and generated $\hat{y}$, computed over voiced frames after DTW alignment. Following MultiVerse.

**F0 RMSE:** Root mean squared error of aligned F0 contours in semitone scale.

### 8.4 Speech Quality

**UTMOS:** Neural MOS predictor score (1–5 scale). Higher = better perceived quality. No human raters needed.

### 8.5 Novel Metrics (Our Contribution)

**Cycle Consistency Gap (CCG):**

$$\text{CCG} = \text{SIM-o}(x, \hat{y}) - \text{SIM-o}(x, \hat{y}')$$

Measures how much additional identity information is lost in the round-trip. Lower CCG = more robust. A system with perfect identity transfer would have $\text{CCG} \approx 0$.

**Compounding Ratio (CR):**

$$\text{CR} = \frac{\text{SIM-o}(x, \hat{y}')}{\text{SIM-o}(x, \hat{y})}$$

Ratio of cycle identity to forward identity. Values close to 1.0 indicate identity is stable through bidirectional transfer. Values significantly below 1.0 indicate compounding degradation.

### 8.6 Statistical Rigor

- Report mean ± standard deviation across speakers for all metrics.
- Conduct paired t-tests or Wilcoxon signed-rank tests for system comparisons (A5 vs each baseline).
- Report statistical significance at $p < 0.05$ with Bonferroni correction for multiple comparisons.
- Bootstrap 95% confidence intervals for primary metrics (SIM-o, WER).

---

## 9. Expected Results Tables (Skeleton)

### Table 1: Main Results — Identity and Intelligibility

| System | EN→ZH SIM-o↑ | EN→ES SIM-o↑ | EN→HI SIM-o↑ | ZH→EN SIM-o↑ | ES→ZH SIM-o↑ | Avg SIM-o↑ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| B1: F5-TTS vanilla | — | — | — | — | — | — |
| B2: XL F5-TTS | — | — | — | — | — | — |
| B3: LoRP-TTS | — | — | — | — | — | — |
| B4: XTTS v2 | — | — | — | — | — | — |
| A1: IAA only | — | — | — | — | — | — |
| A2: IAA + naive GD | — | — | — | — | — | — |
| A3: IAA + meta-opt (no cycle) | — | — | — | — | — | — |
| A4: IAA + meta-opt + cycle (uniform w) | — | — | — | — | — | — |
| **A5: Full CycleAdapt-TTS** | — | — | — | — | — | — |

*(Same table replicated for WER↓, CER↓, F0 PCC↑, UTMOS↑)*

### Table 2: Cycle Consistency Analysis

| System | Avg CCG↓ | Avg CR↑ |
|--------|:---:|:---:|
| B1: F5-TTS vanilla | — | — |
| A1: IAA only | — | — |
| A3: IAA + meta-opt (no cycle) | — | — |
| **A5: Full CycleAdapt-TTS** | — | — |

### Table 3: Prompt Duration Sensitivity (EN→ZH, SIM-o)

| System | 3s | 6s | 10s | 30s |
|--------|:---:|:---:|:---:|:---:|
| B1: F5-TTS vanilla | — | — | — | — |
| B2: XL F5-TTS | — | — | — | — |
| **A5: Full CycleAdapt-TTS** | — | — | — | — |

### Table 4: Zero-Shot Language Pair (ES→ZH)

| System | SIM-o↑ | WER↓ | UTMOS↑ |
|--------|:---:|:---:|:---:|
| B1: F5-TTS vanilla | — | — | — |
| A1: IAA only | — | — | — |
| **A5: Full CycleAdapt-TTS** | — | — | — |

---

## 10. EMNLP Alignment Notes

### 10.1 Why EMNLP?

EMNLP accepts speech + language work, particularly when framed around multilingual/cross-lingual NLP challenges. Relevant EMNLP 2024–2025 papers include VoiceCraft-X (EMNLP 2025 main) and MultiVerse (Findings of EMNLP 2024). Cross-lingual TTS is explicitly within scope when the contribution addresses language transfer, multilingual representation, or language-specific adaptation — all of which apply here.

### 10.2 Framing for EMNLP Reviewers

- **Lead with the language problem, not the audio problem.** Frame as: "How do we preserve speaker identity when transferring across typologically diverse languages?"
- **Emphasize linguistic diversity:** Include tonal (ZH), syllable-timed (ES), and morphologically rich (HI) languages. Discuss phonotactic mismatch as the core challenge.
- **Position cycle consistency as a cross-lingual consistency constraint**, analogous to back-translation in MT — a concept EMNLP reviewers know well.
- **Cite EMNLP/ACL papers heavily:** VoiceCraft-X (EMNLP 2025), MultiVerse (Findings EMNLP 2024), Language-Agnostic Meta-Learning TTS (ACL 2022), ParrotTTS (Findings EACL 2024).

### 10.3 Submission Target

- **EMNLP 2026 deadline:** Typically mid-June 2026 (ARR submission cycle: June 15 commitment deadline, requiring paper submission ~May 15 to ARR).
- **Backup venues:** ICDM 2026 (June deadline), AAAI 2027 (August deadline), NAACL 2026 (if late-breaking deadline available).

### 10.4 Recommended Paper Structure

1. **Introduction** (1 page) — problem, gap, cycle consistency analogy to back-translation, contributions
2. **Related Work** (1 page) — cross-lingual TTS, meta-learning for TTS, test-time adaptation in speech
3. **Method** (2.5 pages) — architecture, losses, meta-training, test-time algorithm
4. **Experimental Setup** (1 page) — datasets, baselines, metrics, hyperparameters
5. **Results** (2 pages) — main results, ablations, convergence, zero-shot, prompt sensitivity, domain shift
6. **Analysis** (0.5 page) — learned weight visualization, computational cost, failure cases
7. **Conclusion** (0.5 page)
8. **Appendix** — full hyperparameter tables, per-speaker breakdowns, additional language pairs

---

## 11. Hyperparameters to Tune

| Hyperparameter | Search Range | Default |
|---------------|-------------|---------|
| Adapter rank $r$ | {4, 8, 16, 32} | 8 |
| Number of adapted DiT layers | {2, 4, 6, all} | 4 |
| Inner steps $K$ (meta-training) | {1, 2, 3, 5} | 3 |
| Inner steps $K$ (test-time) | {1, 2, 3} | 3 |
| Outer loss $\lambda_1$ (CER weight) | {0.1, 0.5, 1.0} | 0.5 |
| Outer loss $\lambda_2$ (F0 weight) | {0.1, 0.3, 0.5} | 0.3 |
| Meta-learning rate $\alpha$ | {1e-4, 3e-4, 1e-3} | 3e-4 |
| Reptile step size $\beta$ | {0.1, 0.3, 0.5} | 0.3 |
| Optimizer MLP hidden dim | {32, 64, 128} | 64 |
| Episode batch size $B$ | {4, 8, 16} | 8 |
| Total meta-iterations $M$ | {5K, 10K, 20K} | 10K |

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Cycle loss is too noisy (round-trip audio quality too low to extract useful signal) | Pre-filter: only backprop cycle loss when UTMOS($\hat{y}$) > threshold; fall back to forward-only losses otherwise |
| Meta-training doesn't converge | Start with Reptile (simpler); if unstable, fall back to first-order MAML with gradient clipping |
| F5-TTS doesn't support Hindi well | Use Cross-Lingual F5-TTS's speaking-rate predictor extension; alternatively substitute MeloTTS or XTTS as backbone |
| Adapter injection into F5-TTS is non-trivial | Start with the simplest injection: LoRA on the final 2 DiT cross-attention layers only; expand if results are promising |
| Speaker encoder gradients don't flow well through frozen model | Use straight-through estimator or detach cycle loss from TTS backward pass and only update adapter via forward loss gradients |

---



---


