# ViT-5 Rearchitecture Proposal

Based on experimental findings from Exp 1 (activation sparsity) and Exp 3 (token diversity & attention entropy).

---

## Problems Identified in ViT-5

| Problem | Source | Evidence |
|---------|--------|----------|
| "Over-gating" mechanism is wrong | Exp 1 | ViT-5 is LESS sparse than DeiT-III (Hoyer 0.51 vs 0.54) |
| LayerScale gammas are not gates | Exp 1 | Range [-57, +83], ~50% negative, std up to 32.5 |
| SwiGLU rejected based on flawed analysis | Exp 1 | Degradation is likely optimization artifact, not sparsity |
| Early-layer over-smoothing | Exp 3 | Token cos-sim 0.66 vs 0.29 at layer 0 |
| More over-smoothed than DeiT-III overall | Exp 3 | Avg token cos-sim 0.49 vs 0.38 |
| Register tokens need fragile RoPE hack | Paper Table 3 | Vanilla registers hurt; fix is uncharacterized |
| No robustness evaluation | Paper | Zero OOD benchmarks for a 2026 backbone paper |

---

## Proposed Changes

### 1. Drop APE, keep only RoPE

**Problem**: The dual APE+RoPE positional encoding causes severe early-layer over-smoothing. Layer 0 token cosine similarity is 0.66 (ViT-5) vs 0.29 (DeiT-III) — a 2.3x difference. Both encodings push nearby patches toward similar representations, over-determining position.

**The paper's justification for keeping APE**: "Patch-level flips become invariant under RoPE-only" (Figure 2 in paper). But this is a contrived edge case — no real image has its 16x16 patches spatially flipped. The over-smoothing cost is real and measured; the flip-invariance cost is hypothetical.

**Change**: Use RoPE-only. Accept the theoretical flip invariance. If flip invariance is genuinely a concern for a downstream task, it can be addressed with data augmentation during fine-tuning.

**Expected effect**: Significantly reduce early-layer over-smoothing. Tokens start more diverse, giving the model more spatial discrimination to work with. This should particularly help dense prediction tasks (detection, segmentation) where per-token differentiation matters.

### 2. Fix LayerScale + SwiGLU instead of abandoning SwiGLU

**Problem**: The paper rejects SwiGLU (the standard FFN in every major LLM) because combining it with LayerScale causes a -0.46% accuracy drop. They attribute this to "over-gating." Our measurements show:
- The over-gating mechanism doesn't exist as described
- LayerScale gammas grow to extreme values [-57, +83] far from their 1e-4 initialization
- The degradation is likely an optimization/initialization artifact

**Change**:
- Initialize LayerScale at **1.0** (identity) instead of 1e-4
- Use SwiGLU as the FFN activation
- Optionally: warmup schedule where LayerScale is frozen at 1.0 for the first N epochs, then unfrozen

**Rationale**: With init=1.0, the FFN contributes normally to the residual stream from the start. SwiGLU's gradients flow cleanly without fighting a near-zero LayerScale. The model can then learn to adjust LayerScale as needed during training, rather than having to climb from 1e-4 to values in the tens/hundreds.

**Expected effect**: Recover SwiGLU's proven expressiveness while maintaining LayerScale's training stability. Aligns vision backbone with LLM practice. The -0.46% degradation should disappear or become a gain.

### 3. Rethink registers

**Problem**: Vanilla registers hurt performance (Table 3: 84.02 -> 83.90 for Base). The paper's fix — giving registers a separate high-frequency RoPE — is ad hoc. The frequency ratio between register RoPE and patch RoPE is never ablated. This is a band-aid over a poorly understood problem.

**Change**: Either:

**(a) Registers without positional encoding** + learned attention bias. The core issue is that unrotated registers have low cosine similarity with nearby rotated patches, distorting attention. Instead of adding another RoPE, add a small learned scalar bias to register-patch attention logits. This is simpler, has one hyperparameter (the init value), and doesn't require reasoning about frequency bases.

**(b) Drop registers entirely.** The paper claims registers suppress attention artifacts (following Darcet et al., 2024). But our Exp 3 shows ViT-5's head diversity is already strong (0.43 vs 0.57 cross-head sim). If registers are primarily serving as attention sinks, a simpler solution is to add a **sink token** — a single fixed token that absorbs excess attention, without the complexity of 4 learnable registers + positional encoding.

### 4. Constrain or replace LayerScale

**Problem**: LayerScale gammas in the trained model range from -57 to +83 with ~50% negative at every layer. This is no longer "a learnable scaling factor initialized to a small value" (the original CaiT design intent). It's an unconstrained signed channel-wise affine transform. The paper's analogy to "static gating" (values in [0,1]) is empirically false.

**Change**: Either:

**(a) Constrain LayerScale** to its intended range. Use `gamma = softplus(raw_gamma)` to keep values positive, or clamp to [0, 2]. This preserves the stabilization intent while preventing the wild growth observed in practice.

**(b) Replace with post-norm.** Table 1 in the paper shows post-normalization gives "highly similar" accuracy to LayerScale across all model sizes (within 0.04%). Post-norm is simpler (no extra parameters), well-understood, and standard in LLMs. It achieves the same stabilization goal without adding a parameter that learns to do something completely different from its design intent.

### 5. QK-Norm with learned temperature

**Problem**: QK-Norm normalizes Q and K to unit norm before the dot product. This stabilizes training (Figure 5 — no loss spikes) but removes magnitude information. All query-key similarities are forced into [-1, 1], and the attention sharpness is controlled solely by sqrt(d). Different heads can't learn different attention sharpness levels.

**Change**: Add a learnable scalar temperature per head:

```
attn = softmax(QK_norm @ K_norm^T / tau_h)
```

where `tau_h` is a learnable scalar per head, initialized to `sqrt(d)`. This preserves QK-Norm's stability benefits while allowing heads to independently control their attention sharpness — some heads can learn sharp (local) attention, others diffuse (global) attention.

**Expected effect**: Better head specialization. Our Exp 3 already shows ViT-5 has good head diversity; this should amplify it while maintaining training stability.

---

## The Proposed Architecture: ViT-5-Fixed

```
Architecture:
  - Patch embedding: standard non-overlapping, linear projection (unchanged)
  - Positional encoding: 2D RoPE only (NO absolute positional embedding)
  - Normalization: RMSNorm (keep — minor but consistent improvement)
  - FFN: SwiGLU (restored, standard LLM practice)
  - Activation scaling: LayerScale init=1.0 (or replace with post-norm)
  - Attention: QK-Norm with learnable per-head temperature
  - Registers: none (or single sink token)
  - QKV bias: none (keep — marginal improvement, structural consistency)
```

### Comparison

| Component | ViT-5 (paper) | ViT-5-Fixed (proposed) | Rationale |
|-----------|---------------|------------------------|-----------|
| Position encoding | APE + 2D RoPE | 2D RoPE only | Fixes early-layer over-smoothing |
| FFN activation | GeLU (SwiGLU rejected) | SwiGLU | Over-gating claim is wrong; fix init instead |
| LayerScale init | 1e-4 | 1.0 (or post-norm) | Prevents optimization conflict with SwiGLU |
| QK-Norm | Standard | + learned temperature | Better head specialization |
| Registers | 4 + high-freq RoPE hack | None (or sink token) | Removes fragile workaround |
| RMSNorm | Yes | Yes | Keep (minor gain, simpler) |
| QKV bias | No | No | Keep (structural consistency) |

### Net changes

- **Fewer components**: 5 design choices vs 7 (removed APE, removed register RoPE hack)
- **Standard LLM alignment**: SwiGLU FFN matches LLaMA/Qwen/Gemma practice
- **No fragile workarounds**: No register-RoPE frequency tuning
- **Addresses measured problems**: Fixes over-smoothing (Exp 3) and enables SwiGLU (Exp 1)

---

## Validation Experiments Needed

To confirm the proposed changes work, the following experiments are needed (all feasible on T4):

1. **LayerScale init=1.0 + SwiGLU**: Train ViT-5-Small for 100 epochs. If accuracy matches or exceeds GeLU variant, the over-gating problem was purely an initialization artifact.

2. **RoPE-only vs APE+RoPE**: Train ViT-5-Small with RoPE only. Measure token cosine similarity — should show much lower early-layer over-smoothing.

3. **Robustness evaluation**: Run Exp 5 on the modified architecture. If reducing over-smoothing improves OOD robustness, it confirms our hypothesis that over-smoothed tokens are less robust.

4. **Dense prediction transfer**: Evaluate on ADE20K. If RoPE-only + SwiGLU improves segmentation (where per-token diversity matters), it validates the architecture changes.

---

## Key Insight

ViT-5 was designed by sequential manual ablation starting from vanilla ViT, adding one component at a time and keeping it if accuracy went up. This greedy approach found a local optimum that:

- Rejects SwiGLU based on a wrong mechanism (over-gating)
- Uses dual positional encoding that causes over-smoothing
- Requires a fragile hack for registers
- Has a component (LayerScale) doing something completely different from its design intent

A better starting point: take the standard LLM transformer recipe (RoPE + SwiGLU + RMSNorm + QK-Norm) and adapt it minimally for vision, rather than starting from the 2020 ViT and manually bolting on components. The proposed architecture does exactly this — it's the LLM recipe with 2D RoPE and learnable temperature, nothing more.
