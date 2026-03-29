# Iterative Visual Self-Correction: A Deep Investigation

Can a model generate an image, "look" at it (using a built-in VLM), identify errors, and re-generate only the faulty parts?

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [Existing Work — Who's Done This?](#2-existing-work)
3. [The Critic: VLMs as Image Judges](#3-the-critic)
4. [The Fixer: Inpainting & Local Editing](#4-the-fixer)
5. [Lessons from Text-Domain Self-Correction](#5-lessons-from-text-domain)
6. [Datasets & Benchmarks](#6-datasets--benchmarks)
7. [Key Open Problems](#7-key-open-problems)
8. [Proposed Architecture](#8-proposed-architecture)
9. [Research Directions to Explore](#9-research-directions)

---

## 1. The Core Idea

The loop:

```
Generate Image (T2I model)
       ↓
Critique Image (VLM identifies errors)
       ↓
Localize Errors (grounding model produces masks)
       ↓
Fix Errors (inpainting model regenerates masked regions)
       ↓
Verify Fix (VLM checks again)
       ↓
Repeat or Accept
```

This is the visual analogue of LLM self-refinement (Madaan et al., 2023), but with unique challenges: the feedback must cross the vision-language boundary twice (image → text critique → image fix), and fixes must be spatially precise.

---

## 2. Existing Work

### 2.1 Full Generate → Evaluate → Fix Loops

**SLD: Self-Correcting LLM-Controlled Diffusion** (Niloy et al., 2024, arXiv 2311.16090)
- The most direct implementation. LLM decomposes prompt into sub-conditions, generates image, LLM evaluates each sub-condition, failed conditions trigger targeted inpainting via Grounding DINO + SAM for masking.
- Results: significant improvement on T2I-CompBench for attribute binding, spatial relationships, and object count.

**Idea2Img** (Yang et al., 2024, arXiv 2310.08541)
- GPT-4V in an iterative self-refinement loop: generate → GPT-4V evaluates → provides text feedback → revises prompt/parameters → regenerate.
- Results: improved quality over 2-3 iterations, but saturates quickly.

**RPG: Recaptioning, Planning, and Generating** (Yang et al., 2024, arXiv 2401.11708)
- Multimodal LLM plans generation by decomposing complex prompts into sub-prompts with spatial layouts, generates sub-images, composes them, and VLM verifies. If verification fails, plan is revised.

**LLM-grounded Diffusion (LMD)** (Lian et al., 2023, arXiv 2305.13655)
- LLM generates spatial layout (bounding boxes) from prompt, guides diffusion model, VLM verifies output, layout revised if needed.

**Mini-DALLE3** (Zeqiang et al., 2023, arXiv 2310.07653)
- LLM rewrites/refines prompt, generates image, VLM evaluates, feedback used to iteratively refine the prompt.

### 2.2 Self-Refining Generation (Implicit Correction)

**CONFORM** (Meral et al., 2024, arXiv 2312.06059)
- Uses contrastive signals during diffusion sampling to self-correct attention maps, ensuring each concept is faithfully represented. Implicit self-refinement during generation.

**SDEdit** (Meng et al., 2021, arXiv 2108.01073)
- Add noise to draft image, denoise with new prompt. Iterative noising-denoising as refinement. Precursor to the self-correction paradigm.

**CONFORM, Attend-and-Excite** (Chefer et al., 2023)
- Ensures all prompt subjects get adequate attention by adding a loss that maximizes minimum attention per subject token. Fixes "missing objects" during generation.

### 2.3 RL/Training-Time Self-Improvement

**DDPO** (Black et al., 2024, arXiv 2305.13301)
- Frames diffusion sampling as multi-step MDP, applies policy gradients with CLIP/VLM reward. Training-time self-correction.

**DPOK** (Fan et al., 2024, arXiv 2305.16381)
- Policy gradient to fine-tune diffusion models with VLM-based reward + KL regularization.

**Diffusion-DPO** (Wallace et al., 2024, arXiv 2311.12908)
- DPO applied to diffusion models. Model generates pairs, preference signal trains it to prefer better outputs. Wins ~70% vs base SDXL.

**ImageReward + RLHF** (Xu et al., 2023, arXiv 2304.05977)
- Reward model trained on human preferences, used for RLHF-style fine-tuning of diffusion models.

### 2.4 Commercial Systems

| System | Self-Correction Type | Critic |
|--------|---------------------|--------|
| **DALL-E 3 + ChatGPT** | Prompt rewriting + selective inpainting (April 2025 canvas editor) | GPT-4V (user-assisted or automatic) |
| **Midjourney Vary (Region)** | Select region → regenerate with new prompt | User acts as critic |
| **Adobe Firefly + Photoshop** | Generative Fill on selected region | User acts as critic |
| **Gemini Image Gen** | Native understanding + generation in one model | Self-critique potential |
| **Stability AI / SD community** | Automated pipelines via ControlNet + inpainting + LLM agents | Programmable |

---

## 3. The Critic: VLMs as Image Judges

### 3.1 Alignment Metrics

| Metric | Paper | Year | ArXiv | Method |
|--------|-------|------|-------|--------|
| CLIPScore | Hessel et al. | 2021 | 2104.08718 | CLIP cosine similarity (coarse, misses composition) |
| TIFA | Hu et al. | 2023 | 2303.11897 | Decompose prompt → VQA questions → answer on image |
| VQAScore | Lin et al. | 2024 | 2404.01291 | P("Yes" \| image, "Does this show {prompt}?") — simple, surprisingly strong |
| DSG | Cho et al. | 2024 | 2310.18235 | Davidsonian scene graph → atomic independent questions |
| LLMScore | Lu et al. | 2023 | 2305.11116 | LLM generates evaluation criteria from prompt |
| GECKO | — | 2024 | 2404.16820 | Prompt → sub-claims → VLM verifies coverage |

### 3.2 Preference/Quality Models

| Model | Paper | Year | ArXiv | Training Data |
|-------|-------|------|-------|---------------|
| ImageReward | Xu et al. | 2023 | 2304.05977 | 137K expert preference pairs, multi-dimensional |
| PickScore | Kirstain et al. | 2023 | 2305.01021 | 500K+ real user preferences (Pick-a-Pic) |
| HPS v2 | Wu et al. | 2023 | 2306.09341 | 798K preference choices, 10+ models |
| Q-Instruct | — | 2024 | 2311.06783 | LLaVA fine-tuned on quality instruction data |
| Prometheus-Vision | — | 2024 | 2401.06591 | VLM trained as rubric-based judge |

### 3.3 Spatial Error Localization

The critical missing piece. Current approaches:

- **Grounding DINO + SAM**: Text → bounding box → precise mask. Works for objects ("left hand"), not for abstract errors ("lighting is wrong").
- **Kosmos-2** (2023, arXiv 2306.14824): Outputs bounding boxes for referred objects. Can localize by asking "where is the red car?"
- **Ferret** (2023, arXiv 2310.07704): Takes spatial inputs (points, boxes, regions) and outputs spatial references.
- **GPT-4V/Gemini with grid overlays**: Ad hoc but shows promise for approximate localization.

**No dedicated system exists for "point to the error in this generated image."** This is the single biggest gap.

### 3.4 Known VLM Limitations as Judges

| Failure Mode | Description | Impact |
|-------------|-------------|--------|
| **Counting blindness** | Struggle with counts > 4-5 | Can't detect "7 fingers" reliably |
| **Spatial confusion** | Left/right, above/below errors | May validate wrong spatial arrangements |
| **Attribute binding** | Can't track which object has which attribute | "Red car, blue truck" → accepts swapped colors |
| **Leniency bias** | Rate images more favorably than humans | Miss subtle errors |
| **Hallucination** | "See" objects mentioned in prompt that aren't present | Inflate alignment scores |
| **Resolution limits** | Process at 336-448px | Miss fine details, small text, subtle artifacts |
| **Negation blindness** | Fail on "no people" | Don't penalize presence of negated objects |
| **Sycophancy** | Rate own outputs higher | Self-critique unreliable |

---

## 4. The Fixer: Inpainting & Local Editing

### 4.1 State-of-the-Art Inpainting Models

| Model | Year | Key Technique | Quality |
|-------|------|---------------|---------|
| **FLUX.1 Fill** | 2024 | Flow-matching transformer inpainting | Highest commercial quality |
| **BrushNet** | 2024 | Plug-in dual-branch for any SD model | Best unmasked preservation |
| **PowerPaint** | 2024 | Task-aware learnable prompts (remove, add, fill) | Most versatile |
| **HD-Painter** | 2024 | Prompt-aware introverted attention | Best text alignment in mask |
| **SDXL Inpainting** | 2023 | Fine-tuned SDXL with mask channel | Widely deployed baseline |

### 4.2 Mask-Guided Regeneration Techniques

| Approach | Paper | Key Idea |
|----------|-------|----------|
| Latent-space masking | Standard SD Inpainting | Replace unmasked latents at each step |
| Blended Diffusion | Avrahami et al. 2022 | Explicit blend: mask * generated + (1-mask) * noised_original |
| **Differential Diffusion** | Levin & Fried 2024 | Continuous change map [0,1] per pixel — partial edits |
| BrushNet injection | Ju et al. 2024 | Inject masked-image features at every UNet layer |

**Best practice**: Feathered masks (10-30px Gaussian blur on edges) + Differential Diffusion or BrushNet for seamless boundaries.

### 4.3 Instruction-Based Editing

| Model | Paper | Year | Limitation |
|-------|-------|------|------------|
| InstructPix2Pix | Brooks et al. | 2023 | No mask → imprecise spatial control |
| MagicBrush | Zhang et al. | 2024 | Better localization (trained on masked edits) |
| MGIE | Fu et al. | 2024 | MLLM derives detailed sub-instructions from vague requests |
| SmartEdit | Huang et al. | 2024 | LLaVA-based understanding of complex edit instructions |

**For critic-fix pipelines: mask-guided inpainting is more reliable than instruction-based editing** for targeted error correction.

### 4.4 Maintaining Global Coherence

| Technique | Purpose |
|-----------|---------|
| DDIM Inversion | Encode original to noise space; edit from that starting point |
| Null-text Inversion (Mokady et al. 2023) | Optimize unconditional embedding for perfect reconstruction + editable |
| Self-attention injection | Inject K, V from original into edited generation for style consistency |
| IP-Adapter | Reference image features via decoupled cross-attention for style matching |
| Image harmonization post-processing | Fix color/lighting mismatches at boundaries |

### 4.5 Multi-Round Editing Degradation

| Issue | Cause | Mitigation |
|-------|-------|------------|
| VAE encode-decode loss | Lossy autoencoder (each round loses info) | Always edit from original latent; use SDXL/FLUX VAE |
| Boundary artifacts | Seams compound | Feathered masks, one-shot multi-region |
| Style drift | Text-guided inconsistencies accumulate | IP-Adapter referencing original |

**Practical limit: 3-4 rounds** without careful pipeline design. Beyond that, quality degrades noticeably.

### 4.6 Text → Mask Pipeline

```
Critic Output (e.g., "6 fingers on left hand")
        ↓
Grounding DINO → bounding box for "left hand"
        ↓
SAM 2 → precise binary mask
        ↓
Dilate + Feather (10-30px)
        ↓
BrushNet / FLUX Fill → inpaint with fix prompt
        ↓
(Optional) Image Harmonization
```

---

## 5. Lessons from Text-Domain Self-Correction

### 5.1 The Core Debate: Does Self-Correction Work?

**It works WITH external grounding. It fails WITHOUT it.**

| Paper | Year | Key Finding |
|-------|------|-------------|
| **Self-Refine** (Madaan et al.) | 2023 | Same LLM can generate → critique → refine; +5-40% improvement across 7 tasks |
| **Reflexion** (Shinn et al.) | 2023 | Verbal self-reflections stored in memory improve subsequent attempts (91% on HumanEval) |
| **"LLMs Cannot Self-Correct Reasoning Yet"** (Huang et al.) | 2024 | **Without external feedback**, self-correction degrades performance. Flips correct→incorrect as often as incorrect→correct |
| **Stechly et al.** | 2024 | GPT-4 self-verification only slightly better than chance on planning |
| **Tyen et al.** | 2024 | Self-correction fixes sampling errors, not capability gaps |
| **Kamoi et al.** | 2024 | Works on automatically verifiable tasks (code, math); fails on open-ended tasks |

**Visual implication**: A VLM critiquing its own generated image is self-correction WITHOUT external grounding (same model, same blind spots). You NEED external signals: object detectors, OCR, layout analyzers, specialized critics.

### 5.2 Key Principles from Text Domain

| Principle | Source | Visual Implication |
|-----------|--------|-------------------|
| External grounding is essential | Huang et al. 2024 | Use object detectors, OCR, pose estimators — not just VLM self-critique |
| Verification is easier than generation | Lightman et al. 2023 | VLMs can reliably critique even when they can't generate perfectly |
| Dedicated critics > self-critique | Shepherd (Wang et al. 2023), CriticGPT | Train a specialized visual critic, don't rely on generator self-assessment |
| Process supervision > outcome supervision | Lightman et al. 2023 | Critique each image region/element, not just the whole image |
| Refinement saturates in 2-4 iterations | Madaan et al. 2023, Idea2Img | Don't expect infinite improvement; design for 2-3 cycles |
| Fixes execution errors, not capability gaps | Tyen et al. 2024 | Can fix "wrong finger count" but not "can't render complex 3D" |
| Best-of-N + verification is a strong baseline | Snell et al. 2024 | Generate N images, pick best BEFORE attempting targeted edits |
| Multi-agent critique catches blind spots | Du et al. 2023 | Multiple specialized critics > one general critic |
| Constitutional principles provide grounding | Bai et al. 2022 | Define checkable visual rules ("hands have 5 fingers") |

### 5.3 The "Verifier Is Easier Than Generator" Principle

This is the theoretical foundation for why visual self-correction should work:

- **Lightman et al. 2023** (OpenAI): Process reward models verifying each step boost math accuracy from 49% → 78%
- **Snell et al. 2024**: Given fixed compute, generate N + verify is better than single expensive generation
- **Complexity theory**: Generation is "NP-like" (many possible outputs), verification is "P-like" (checking a specific image is easier)

**For images**: Generating a perfect hand is hard. Checking whether a hand has 5 fingers is easy. This asymmetry is the fundamental reason the loop should work.

### 5.4 Tree/Graph of Thoughts for Images

| Method | Paper | Year | Visual Analogue |
|--------|-------|------|-----------------|
| Tree of Thoughts | Yao et al. | 2023 | Generate multiple candidates at each refinement step, pursue best branches |
| Graph of Thoughts | Besta et al. | 2024 | Merge layout from branch A + style from branch B |
| LATS | Zhou et al. | 2024 | Monte Carlo tree search over edit sequences |

---

## 6. Datasets & Benchmarks

### 6.1 Text-to-Image Alignment Benchmarks

| Dataset | Size | What It Tests | Availability |
|---------|------|---------------|--------------|
| T2I-CompBench | 6K prompts | Attribute binding, spatial, counting | Public |
| T2I-CompBench++ | ~8K prompts | + numeracy, complex compositions | Public |
| TIFA | 4K prompts, 25K QA | Element-level faithfulness via VQA | Public |
| DSG | 1,060 prompts | Atomic independent questions with dependency graph | Public |
| GenAI-Bench | 1,600 prompts | Compositional alignment + human ratings | Public |
| DPG-Bench | 1,065 prompts | Dense/long prompts (~67 words avg) | Public |
| PartiPrompts | 1,632 prompts | 11 categories, broad coverage | Public |
| DrawBench | 200 prompts | Color, counting, spatial, text rendering | Public |
| HRS-Bench | 13K prompts | 13 skills including fairness, robustness | Public |
| CountBench | ~500 prompts | Counting accuracy (1-10 objects) | Public |
| ABC-6K | 6,400 prompts | Attribute binding + counting | Public |

### 6.2 Human Preference Datasets

| Dataset | Size | Annotation Type | Availability |
|---------|------|-----------------|--------------|
| Pick-a-Pic | 500K-1M pairs | Binary preference from real users | Public (HuggingFace) |
| HPD v2 | 798K choices | Preference across 4 styles, 10+ models | Public |
| ImageRewardDB | 137K pairs | Expert multi-dimensional scores | Public |
| **RichHF-18K** | 18K images | **Pixel-level error heatmaps + NL descriptions** | Public (Google Research) |
| AGIQA-3K | 2,982 images | MOS quality + alignment scores | Public |
| AIGCIQA2023 | 2,400 images | Quality, authenticity, correspondence | Public |

**RichHF-18K is the most relevant** for self-correction — it has pixel-level implausibility heatmaps, artifact annotations, misalignment regions, and natural language error descriptions.

### 6.3 Editing Benchmarks

| Dataset | Size | Key Feature | Availability |
|---------|------|-------------|--------------|
| **MagicBrush** | 10K turns, 5K sessions | Multi-turn + manual ground truth + masks | Public |
| EditBench | 240 triples | Text-guided inpainting evaluation | Partial |
| InstructPix2Pix | 454K train | Synthetic instruction editing | Public |
| Emu Edit | 535 test | 7 operation types | Public |
| EditVal | 648 operations | 19 edit types | Public |
| HQ-Edit | 200K pairs | GPT-4V + DALL-E 3 generated | Public |

**MagicBrush is the most relevant** for multi-turn self-correction — 1-3 sequential edits per session with intermediate results and ground truth.

### 6.4 Synthetic Data Generation for Training Critics

| Approach | Method | Output |
|----------|--------|--------|
| LLM error descriptions | GPT-4V critiques generated images | (image, error_text, region_text) |
| Prompt perturbation | Swap colors/counts/spatial in correct pairs | Negative examples with known errors |
| Image perturbation | Inpaint controlled errors into correct images | (image, error_mask, error_type) |
| Scene graph comparison | Detected vs intended scene graph | Structural error annotations |
| Layout comparison | Object detection vs intended layout | Bounding box error localization |
| VQA error mining | Incorrectly answered questions → errors | Per-element error identification |
| Reward model bootstrapping | Generate N, rank, top/bottom pairs | Preference data at scale |

---

## 7. Key Open Problems

### 7.1 The Spatial Localization Gap

**No system can reliably go from "the hand is wrong" to a pixel-precise mask of the hand.** Grounding DINO + SAM works for nameable objects but fails for:
- Abstract errors ("the lighting is inconsistent")
- Relational errors ("the shadow goes the wrong direction")
- Partial errors ("the third finger from the left is bent wrong")
- Stylistic errors ("this region looks like a different art style")

### 7.2 The Critique-Fix Information Bottleneck

The loop crosses the vision-language boundary twice:
```
Image → (VLM) → Text critique → (T2I) → Fixed image
```
Each crossing loses information. The text critique can't fully specify the visual fix. "Fix the hand" doesn't tell the inpainting model the exact pose, skin tone, lighting angle, or finger positions.

### 7.3 Shared Blind Spots

VLMs and T2I models often share the same weaknesses (counting, spatial reasoning, fine-grained text). If the critic has the same blind spots as the generator, the loop doesn't help. This is the visual equivalent of Huang et al.'s finding that LLMs cannot self-correct without external grounding.

### 7.4 Coherence Degradation Over Rounds

Each edit introduces subtle inconsistencies (style drift, boundary artifacts, VAE loss). After 3-4 rounds, the image may be globally worse even if each local fix was correct.

### 7.5 When to Stop

No reliable criterion for "this image is now good enough." VLMs are lenient judges (Section 3.4). The system may oscillate: fix A breaks B, fix B breaks C, fix C breaks A.

### 7.6 Capability Gaps vs Execution Errors

Self-correction can fix execution errors (the model "knows" how to draw hands but got unlucky in this sample) but cannot fix capability gaps (the model fundamentally can't render certain compositions). The system needs to distinguish these and bail out on capability gaps rather than entering infinite correction loops.

---

## 8. Proposed Architecture

### 8.1 The Full Pipeline

```
┌─────────────────────────────────────────────────┐
│                   GENERATOR                       │
│  T2I Model (FLUX / SD3 / DALL-E)                 │
│  Input: text prompt + (optional) layout          │
│  Output: candidate image                          │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│              CRITIC ENSEMBLE                      │
│  1. VLM Judge (GPT-4V / Qwen-VL / LLaVA-Critic)│
│     → overall assessment + error descriptions    │
│  2. Object Counter (specialized)                  │
│     → count verification                          │
│  3. OCR Engine (if text in prompt)                │
│     → text rendering verification                 │
│  4. Pose Estimator (if humans in prompt)          │
│     → anatomical plausibility                     │
│  5. Layout Verifier (detected vs intended boxes)  │
│     → spatial relationship check                  │
│                                                   │
│  Output: {pass: bool, errors: [{desc, region,    │
│           severity, fix_prompt}]}                 │
└────────────────────┬────────────────────────────┘
                     ↓ (if errors found)
┌─────────────────────────────────────────────────┐
│              ERROR LOCALIZER                      │
│  Grounding DINO → bounding box per error region  │
│  SAM 2 → precise segmentation mask               │
│  Dilate + Gaussian feather (10-30px)             │
│  Output: soft mask per error                      │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│              FIXER                                │
│  Strategy selection:                              │
│  - Minor fix → Differential Diffusion (low str)  │
│  - Region fix → BrushNet / FLUX Fill (masked)    │
│  - Major fix → full regeneration with new seed   │
│  + IP-Adapter referencing original for style      │
│  + Combined mask for multiple errors (one pass)   │
│  Output: fixed image                              │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│              VERIFIER                             │
│  Same critic ensemble on fixed image              │
│  Compare scores: improved? degraded? new errors? │
│  Decision: accept / retry fix / full regenerate  │
│  Max iterations: 3                                │
└─────────────────────────────────────────────────┘
```

### 8.2 Key Design Decisions

**Best-of-N first, then targeted correction.** Generate N=4 candidates, pick the best via VLM ranking (cheap), THEN apply the correction loop only if the best candidate still has errors. This avoids wasting expensive correction compute on bad starting points.

**Critic ensemble, not single VLM.** Use specialized tools (object counter, OCR, pose estimator) alongside a general VLM. This provides the external grounding that pure self-critique lacks.

**One-shot multi-region fix.** If multiple errors are detected, combine all masks and fix in a single inpainting pass. This avoids multi-round degradation.

**Severity-based strategy.** Minor issues (slight color off) → low-strength Differential Diffusion. Moderate issues (wrong object attribute) → masked inpainting. Major issues (completely wrong composition) → full regeneration with modified prompt.

**Hard stop at 3 iterations.** Evidence from both text (Madaan et al.) and visual (Idea2Img) domains shows refinement saturates at 2-3 rounds. After that, return the best candidate seen so far.

---

## 9. Research Directions to Explore

### 9.1 Unified Critic-Generator Models

Instead of separate VLM + T2I models, train a single multimodal model that can both generate and critique. Gemini's architecture (native multimodal) hints at this. The model generates, then in a second forward pass "looks" at its own output and identifies errors, then produces targeted edits — all within one model. This eliminates the text bottleneck between critic and fixer.

**Key question**: Does self-critique work better when the critic and generator share weights (same model understands its own failure modes) or worse (shared blind spots)?

### 9.2 Process Supervision for Diffusion

Instead of critiquing the final image, critique intermediate denoising steps. A "process reward model" for diffusion could evaluate at t=500, t=250, t=100 and intervene early when things go wrong — before errors become baked into the image. Analogous to Lightman et al.'s step-level math verification.

### 9.3 Visual Constitution

Define a formal set of checkable visual rules:
- "Human hands have exactly 5 fingers"
- "Text matches the prompt exactly"
- "Shadows are consistent with a single light source"
- "Object counts match the prompt"
- "Named colors match the prompt"

Each rule maps to a specialized checker. The constitution provides the external grounding that makes self-correction work (analogous to Anthropic's Constitutional AI).

### 9.4 Error-Aware Generation

Train the generator to be aware that it will be critiqued and corrected. Self-Refine with Refinement-Aware Training (Ye et al., 2024) showed that text models trained with this awareness produce better initial outputs AND respond better to feedback. Applied to diffusion: the model might learn to produce images that are more amenable to targeted edits — e.g., cleaner boundaries around objects, more compositionally modular layouts.

### 9.5 Learned Error Localization

Train a model specifically for "find the error and produce a mask." Input: (image, prompt, error_description). Output: segmentation mask of the error region. Training data from RichHF-18K (pixel-level error annotations) + synthetic perturbation datasets. This fills the spatial localization gap (Section 7.1).

### 9.6 Diffusion Beam Search

Run multiple denoising trajectories in parallel, evaluate intermediates with a lightweight critic, and allocate more compute to promising trajectories. A form of test-time compute scaling for images. Related to Snell et al.'s "Scaling LLM Test-Time Compute Optimally."

### 9.7 Multi-Agent Visual Debate

Multiple VLMs (or the same VLM with different prompts/personas) critique an image from different angles:
- Agent A: anatomical correctness specialist
- Agent B: compositional/spatial specialist
- Agent C: aesthetic/style specialist
- Agent D: text-rendering specialist

Disagreements between agents flag uncertain regions for human review or conservative re-generation. Multi-agent debate outperforms single-agent self-critique in text (Du et al., 2023).

### 9.8 Reward Model for Iterative Editing Quality

Train a reward model not just on single-image quality but on edit quality: given (original_image, edit_mask, edited_image, prompt), predict whether the edit improved the image without introducing new problems. This captures the multi-round degradation issue and could serve as the stop criterion.

### 9.9 Benchmark: Self-Correction Challenge

No benchmark exists specifically for evaluating self-correction systems. Propose one:
- 500 prompts where single-shot generation consistently fails (hard compositional, counting, spatial)
- Evaluate: (a) single-shot accuracy, (b) accuracy after 1 correction round, (c) after 2 rounds, (d) after 3 rounds
- Measure: improvement per round, coherence degradation per round, compute cost per round
- Track: which error types are fixable vs unfixable (capability gap detection)

---

## Key Papers Reference Table

| Paper | Year | ArXiv | What It Does |
|-------|------|-------|--------------|
| SLD | 2024 | 2311.16090 | Full generate→evaluate→inpaint loop |
| Idea2Img | 2024 | 2310.08541 | GPT-4V iterative refinement |
| RPG | 2024 | 2401.11708 | Plan→generate→verify with MLLM |
| Self-Refine | 2023 | 2303.17651 | LLM self-correction (text domain foundation) |
| "LLMs Cannot Self-Correct" | 2024 | 2310.01798 | Self-correction needs external grounding |
| Lightman et al. (PRM) | 2023 | 2305.20050 | Process supervision > outcome supervision |
| VQAScore | 2024 | 2404.01291 | Best simple alignment metric |
| ImageReward | 2023 | 2304.05977 | Trained preference model |
| BrushNet | 2024 | — | Best plug-in inpainting |
| Differential Diffusion | 2024 | — | Continuous-strength local editing |
| Grounding DINO + SAM 2 | 2023-24 | — | Text→mask pipeline |
| RichHF-18K | 2024 | 2312.10240 | Pixel-level error annotations |
| MagicBrush | 2024 | — | Multi-turn editing dataset |
| DDPO | 2024 | 2305.13301 | RL for diffusion (training-time) |
| Diffusion-DPO | 2024 | 2311.12908 | DPO for diffusion |
| Shepherd | 2023 | — | Dedicated critic > self-critique |
| Tree of Thoughts | 2023 | 2305.10601 | Multi-path search with evaluation |
| Attend-and-Excite | 2023 | — | Fix missing objects during generation |
| CONFORM | 2024 | 2312.06059 | Self-correcting attention during diffusion |
| LLaVA-Critic | 2024 | — | VLM fine-tuned for visual critique |
