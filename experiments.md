# ViT-5 Critique: Experiment Plan

All experiments designed for Colab free tier (T4 16GB).

---

## Phase 1: Forward-Pass-Only Analyses (No Training, ~2-4 hours)

### Exp 1 — Activation Sparsity Analysis
- **Tests:** "Over-gating" claim (Section 3.3 of ViT-5)
- **Method:** Measure Hoyer sparsity, Gini coefficient, % dead neurons in FFN activations at every layer
- **Compare:** ViT-5 (GeLU) vs DeiT-III vs vanilla ViT+SwiGLU vs ViT-5+SwiGLU
- **Cost:** ~30 min on T4

### Exp 2 — Effective Rank per Layer
- **Tests:** "Improved representations" claim
- **Method:** SVD of token representation matrix at each layer, compute effective rank = exp(entropy of normalized singular values)
- **Compare:** ViT-5 vs DeiT-III across all layers
- **Cost:** ~1 hour

### Exp 3 — Token Diversity & Attention Entropy
- **Tests:** Over-smoothing / attention collapse
- **Method:** Avg pairwise cosine similarity between tokens per layer, Shannon entropy of attention weights per head/layer, cross-head attention similarity
- **Cost:** ~30 min

### Exp 4 — CKA Heatmaps
- **Tests:** Whether ViT-5 learns fundamentally different features
- **Method:** Intra-model and cross-model linear CKA, subsample ~5K images
- **Cost:** ~1 hour

---

## Phase 2: Robustness Evaluation (No Training, ~3-5 hours)

### Exp 5 — OOD Benchmark Suite
- **Tests:** Robustness (total blind spot in paper)
- **Benchmarks:** ImageNet-C (15 corruptions x 5 severities), ImageNet-A, ImageNet-R, ImageNet-Sketch
- **Cost:** ~3-4 hours

### Exp 6 — Shape vs Texture Bias
- **Tests:** Whether register tokens + RoPE improve shape bias
- **Method:** Geirhos et al. cue-conflict stimuli
- **Cost:** ~30 min

---

## Phase 3: Component Interaction Study (Requires Training, ~2-3 days)

### Exp 7 — Fractional Factorial Design (ViT-5-Small)
- **Tests:** Whether 7 components interact (only 1/21 pairwise interactions tested in paper)
- **Method:** 2^{7-3} Resolution IV design = 16 runs, 100-epoch short schedule on ImageNet
- **Tool:** pyDOE2.fracfact() for design matrix, fANOVA for analysis
- **Cost:** ~2-3 days

### Exp 8 — "Just Scale Up Vanilla" Baseline
- **Tests:** Whether a wider vanilla ViT matches ViT-5-B at same FLOPs
- **Method:** Vanilla ViT-B with hidden dim 880 (matching ViT-5-B FLOPs)
- **Cost:** ~6-8 hours

---

## Phase 4: Probing & Transfer (~1-2 days)

### Exp 9 — Layer-wise Linear Probing
- **Tests:** Feature hierarchy quality
- **Method:** Linear classifier on frozen features from every layer
- **Cost:** ~2-3 hours

### Exp 10 — Few-Shot Transfer
- **Tests:** Whether +0.4% ImageNet gain transfers
- **Method:** k-NN (k=20) on frozen features for CUB-200, Stanford Cars, Oxford Flowers
- **Cost:** ~1-2 hours

### Exp 11 — Loss Landscape Sharpness
- **Tests:** "Stabilization" claim
- **Method:** Top-5 Hessian eigenvalues via PyHessian
- **Cost:** ~2-3 hours per model

---

## Priority

| Phase | Experiments | T4 Time | What It Tests |
|-------|-----------|---------|---------------|
| 1 | Sparsity, Rank, CKA, Entropy | ~4 hrs | Representation & over-gating claims |
| 2 | IN-C/A/R/Sketch, Shape bias | ~4 hrs | Robustness (missing from paper) |
| 3 | 16-run factorial, wider ViT | ~3 days | Component interactions & complexity justification |
| 4 | Probing, few-shot, Hessian | ~1 day | Feature quality & loss landscape |
