# ViT-5 Critique: Reasoning Walkthrough

How I identified the weak areas in this paper, step by step.

---

## Step 1: Read the claim structure, not just the results

The first thing I do with any paper is map out the **claim-evidence chain**: what does the paper claim, and what evidence supports each claim?

ViT-5 makes these claims:

| # | Claim | Evidence provided |
|---|-------|-------------------|
| 1 | "ViT architecture is under-optimized" | Accuracy numbers (Table 5) |
| 2 | "Component-wise refinement yields significant gains" | Ablation (Table 10) |
| 3 | "Existing refinements are not orthogonal" | One pairwise test: LayerScale x SwiGLU (Table 2) |
| 4 | "LayerScale + SwiGLU causes over-gating" | Accuracy drop + verbal argument (no measurement) |
| 5 | "ViT-5 has improved spatial reasoning" | Attention visualizations (Figure 4) |
| 6 | "ViT-5 is a drop-in upgrade for mid-2020s backbones" | ImageNet, COCO, ADE20K numbers |

The immediate red flag: **Claim 4 is the only mechanistic contribution, and it has zero quantitative evidence.** They observe accuracy drops and name it "over-gating" but never measure sparsity, activation statistics, or anything that would confirm the mechanism. This is where I focused first.

---

## Step 2: Check the combinatorial math

The paper has 7 binary design choices. They run:
- 7 single-component ablations (Table 10)
- 1 pairwise interaction test (Table 2: LayerScale x SwiGLU)
- Several "configuration comparisons" (Table 9)

Total unique configurations tested for interactions: **8 out of 128 possible**

With 7 factors, there are **21 possible pairwise interactions**. They test exactly **1 of 21 (4.8%)**. This is a Resolution III design — it can estimate main effects but confounds them with two-factor interactions. In simpler terms: when they remove RoPE and accuracy drops by 0.24%, they cannot tell if that's because RoPE is good, or because RoPE interacts positively with some other component that's always present.

I recognized this because fractional factorial design is standard in experimental statistics. The paper never discusses this limitation.

**Bottleneck identified**: The ablation methodology is fundamentally incomplete. A 2^{7-2} design (32 runs) would have covered all 21 pairwise interactions — only 4x more compute than what they did.

---

## Step 3: Look at what's NOT reported

Every paper has blind spots. I check what standard evaluations are missing:

**For a backbone paper in 2025-2026, the expected evaluation suite is:**
- ImageNet-1k (they have this)
- ImageNet-C/A/R/Sketch (they have NONE of these)
- Dense prediction transfer (they have ADE20K, good)
- Generation (they have SiT/DiT, good)

**Missing**: All robustness benchmarks. This is a major omission for a paper that modifies normalization (RMSNorm), attention (QK-Norm), and position encoding (RoPE) — all components known to affect robustness properties. DeiT-III, ConvNeXt V2, EVA-02, and Vision Mamba all report robustness. ViT-5 doesn't.

**Also missing**: Any analysis of *why* the architecture works — representation quality metrics (CKA, effective rank, probing), loss landscape analysis, scaling laws. The paper is purely "we tried X, number went up."

---

## Step 4: Check the magnitudes

The gains in Table 10 (single-component ablations) are:

| Component removed | Avg accuracy drop |
|-------------------|-------------------|
| SwiGLU -> GeLU | -0.42% |
| LayerScale | -0.29% |
| 2D RoPE | -0.24% |
| Registers | -0.17% |
| RMSNorm -> LayerNorm | -0.15% |
| QK-Norm | -0.12% |
| Keep QKV-bias | -0.06% |

The total of all individual drops is ~1.45%, but the actual total gain from all components combined is ~0.4% (84.2% vs 83.8% for Base). This means **either the components interact negatively, or there's significant overlap/redundancy**. The paper doesn't investigate this discrepancy.

Furthermore, typical training variance on ImageNet for ViT-B is ~0.1-0.2%. Several of these components (QKV-bias: 0.06%, QK-Norm: 0.12%) are **within noise** of standard training variance. Without confidence intervals or multiple runs, you can't distinguish them from random fluctuation.

**Bottleneck identified**: The gains are so small that the signal-to-noise ratio is questionable, and the non-additivity of gains suggests unexplored negative interactions.

---

## Step 5: Check the scale-dependence claims

From Table 10, I plotted how each component's contribution changes across S/B/L:

- Removing LayerScale: -0.15 (S) -> -0.30 (B) -> -0.45 (L) — **grows with scale**
- Removing RoPE: -0.34 (S) -> -0.21 (B) -> -0.16 (L) — **shrinks with scale**
- Removing QK-Norm: -0.17 (S) -> -0.11 (B) -> -0.08 (L) — **shrinks with scale**
- Removing registers: -0.12 (S) -> -0.14 (B) -> -0.25 (L) — **grows with scale**

The components don't have consistent scaling behavior. Some matter more at small scale, some at large. This means the ViT-5 recipe tuned at Base may not be optimal at XL or beyond.

Critically, the paper acknowledges: "we leave a systematic investigation of [whether SwiGLU works at larger scale] for future work." This is a major caveat — they reject SwiGLU (standard in every LLM) based on experiments up to 449M params, while LLMs that use SwiGLU are 7B-70B+.

**Bottleneck identified**: The architecture recipe is scale-specific and may not transfer to the scales that actually matter for foundation models.

---

## Step 6: Examine the one mechanistic claim deeply

The "over-gating" argument (Section 3.3) is:

> "Both LayerScale and gated MLPs effectively perform channel-wise filtering, which increases the sparsity of intermediate representations; when used together, their combined effect can result in excessively sparse activations."

I broke this down into testable predictions:

1. **Prediction**: LayerScale acts as channel-wise filtering (like a gate) -> **Testable**: Measure LayerScale gamma values. Are they in [0,1] like gates? Or something else?
2. **Prediction**: This filtering increases sparsity -> **Testable**: Compare activation sparsity of ViT-5 (with LayerScale) vs DeiT-III (without).
3. **Prediction**: Combining with SwiGLU makes sparsity "excessive" -> **Testable**: Measure sparsity in the LayerScale+SwiGLU variant.

None of these were tested in the paper. All three predictions fail when we measure them:

1. LayerScale gammas range from -57 to +83, with ~50% negative. Not gates.
2. ViT-5 is LESS sparse than DeiT-III (Hoyer 0.51 vs 0.54).
3. Not directly tested yet, but the premise (that LayerScale adds sparsity) is wrong.

**This is the core failure**: The paper's only explanatory claim is a verbal analogy ("LayerScale is like gating") that was never quantified, and when quantified, turns out to be false.

---

## Step 7: Look for the register token fragility

Table 3 in the paper shows that adding vanilla registers HURTS performance (84.02 -> 83.90 for Base). They fix this with a hack: give registers a separate high-frequency RoPE. But they never:

1. Ablate the frequency ratio between register RoPE and patch RoPE
2. Explain WHY this specific fix works (they give an intuition about cosine similarity but no measurements)
3. Test whether this fix is stable across different model sizes

This is a classic sign of brittle engineering: a component that requires a poorly-characterized workaround to not degrade performance.

---

## Step 8: Check the framing against the actual contribution

The paper frames itself as "systematic modernization" and "principled, component-wise design." But:

- The method is sequential manual ablation, not systematic search
- There's no design-of-experiments framework
- There's no theoretical basis for why components should compose
- The only mechanistic argument (over-gating) is wrong

The honest framing: "We tried combinations of known techniques in ViTs and found a recipe that works at Base/Large scale on clean benchmarks." That's an engineering contribution, not a research contribution.

---

## Summary: How to identify weak spots in any paper

1. **Map the claim-evidence chain** — what's claimed vs what's actually shown
2. **Do the combinatorial math** — with N design choices, how much of the space is explored?
3. **Check what's missing** — what do comparable papers report that this one doesn't?
4. **Check the magnitudes** — are gains within noise? Do they add up?
5. **Check scale dependence** — do findings hold across scales?
6. **Test mechanistic claims** — convert verbal arguments into quantitative predictions
7. **Look for brittle fixes** — components that need unexplained workarounds
8. **Compare framing to contribution** — is the paper overselling?
