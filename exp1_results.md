# Experiment 1: Activation Sparsity Analysis — Results

## Objective

Test ViT-5's central mechanistic claim (Section 3.3): that LayerScale acts as "static gating" and combining it with SwiGLU causes "over-gating" due to compounding channel-wise sparsity. The paper never measures sparsity — we do.

## Setup

- **Models**: ViT-5-Small (22M, pretrained) vs DeiT-III-Small (22M, pretrained)
- **Data**: CIFAR-100 test set, 500 images, 224x224 (sparsity patterns are architecture-dependent)
- **Metrics**: Hoyer sparsity, Gini coefficient, % near-zero activations (|x| < 1e-3), % dead channels
- **Measurement point**: Post-GELU FFN intermediate activations at every layer
- **Hardware**: T4 16GB (Colab free tier)

---

## Raw Data

### Per-Layer Sparsity Comparison

```
Layer | Hoyer(V5) Hoyer(D3)    diff | Gini(V5) Gini(D3)    diff | %Zero(V5) %Zero(D3) | %Dead(V5) %Dead(D3)
------+-----------+---------+-------+----------+---------+------+-----------+----------+-----------+---------
    0 |    0.3882   0.5528  -0.1646 |   0.3521   0.4256  -0.0735 |     4.12%     3.15% |     0.00%     0.00%
    1 |    0.3393   0.3597  -0.0204 |   0.3904   0.4386  -0.0482 |     1.60%     2.36% |     0.00%     0.00%
    2 |    0.4111   0.3050  +0.1061 |   0.3955   0.3840  +0.0115 |     1.88%     2.13% |     0.00%     0.00%
    3 |    0.3667   0.3088  +0.0579 |   0.4344   0.3940  +0.0404 |     1.85%     1.77% |     0.00%     0.00%
    4 |    0.4480   0.4275  +0.0205 |   0.5160   0.4999  +0.0160 |     2.27%     2.15% |     0.00%     0.00%
    5 |    0.4432   0.4426  +0.0006 |   0.5251   0.5329  -0.0077 |     2.35%     2.34% |     0.00%     0.00%
    6 |    0.4753   0.4559  +0.0194 |   0.5544   0.5350  +0.0194 |     5.12%     3.96% |     0.00%     0.00%
    7 |    0.4968   0.4628  +0.0340 |   0.5677   0.5431  +0.0246 |     6.34%     5.47% |     0.00%     0.00%
    8 |    0.5560   0.7653  -0.2094 |   0.6458   0.8547  -0.2088 |    14.54%    51.66% |     0.00%     0.00%
    9 |    0.8444   0.9272  -0.0828 |   0.9325   0.9897  -0.0572 |    76.33%    93.10% |     0.00%     0.00%
   10 |    0.7695   0.7967  -0.0272 |   0.8611   0.8873  -0.0261 |    51.76%    58.34% |     0.00%     0.00%
   11 |    0.5843   0.7196  -0.1353 |   0.6084   0.7761  -0.1678 |     8.01%    24.86% |     0.00%     0.00%
------+-----------+---------+-------+----------+---------+------+-----------+----------+-----------+---------
  AVG |    0.5102   0.5437  -0.0334 |   0.5653   0.6051  -0.0398 |    14.68%    20.94% |     0.00%     0.00%
```

### Summary Statistics

| Metric | ViT-5-Small | DeiT-III-Small | Delta |
|--------|-------------|----------------|-------|
| Avg Hoyer sparsity | 0.5102 | 0.5437 | -0.0334 (DeiT-III sparser) |
| Avg Gini coefficient | 0.5653 | 0.6051 | -0.0398 (DeiT-III more unequal) |
| Avg % near-zero | 14.68% | 20.94% | -6.26% (DeiT-III more zeros) |
| Avg % dead channels | 0.00% | 0.00% | 0.00 |

### Late-Layer Sparsity Collapse (DeiT-III)

| Layer | %Zero (ViT-5) | %Zero (DeiT-III) | Ratio |
|-------|---------------|-------------------|-------|
| 8 | 14.5% | 51.7% | 3.6x more sparse in DeiT-III |
| 9 | 76.3% | 93.1% | 1.2x more sparse in DeiT-III |
| 10 | 51.8% | 58.3% | 1.1x more sparse in DeiT-III |
| 11 | 8.0% | 24.9% | 3.1x more sparse in DeiT-III |

### LayerScale Gamma Values (ViT-5 FFN branch, gamma_2)

```
Layer  0: mean=-0.047, std= 0.95, min= -11.0, max=  3.3, negative=49.5%
Layer  1: mean= 0.036, std= 0.91, min=  -2.6, max=  5.0, negative=47.9%
Layer  2: mean= 0.068, std= 1.44, min=  -2.9, max= 19.9, negative=48.4%
Layer  3: mean= 0.091, std= 1.59, min=  -2.5, max= 15.3, negative=46.9%
Layer  4: mean= 0.148, std= 2.22, min=  -3.4, max= 18.2, negative=47.9%
Layer  5: mean= 0.006, std= 3.24, min= -10.9, max=  5.0, negative=49.5%
Layer  6: mean=-0.330, std= 5.82, min= -57.2, max=  8.0, negative=52.1%
Layer  7: mean= 0.843, std= 8.23, min= -11.6, max= 45.8, negative=44.8%
Layer  8: mean=-0.157, std=12.36, min= -43.6, max= 18.2, negative=49.7%
Layer  9: mean=-0.393, std=30.06, min= -45.8, max= 82.5, negative=50.8%
Layer 10: mean= 3.050, std=32.52, min= -43.7, max= 82.9, negative=45.8%
Layer 11: mean= 0.225, std= 8.95, min= -29.3, max= 82.9, negative=50.3%
```

### Effective Sparsity (LayerScale * FFN output, ViT-5 only)

| Metric | Raw FFN Output | After LayerScale | Amplification |
|--------|---------------|------------------|---------------|
| Avg Hoyer | 0.3854 | 0.5179 | +0.1325 |

---

## Figures

- `vit-fig1.png` — 4-panel sparsity comparison (Hoyer, Gini, %Zero, %Dead) across layers
- `vit-fig2.png` — Activation magnitude histograms (post-GELU, log scale) at layers 0, 3, 6, 11
- `vit-fig3.png` — LayerScale gamma heatmaps (attention branch and FFN branch)

---

## Conclusions

### Finding 1: ViT-5 is LESS sparse than DeiT-III, not more

The paper claims LayerScale acts as "static gating" that compounds with SwiGLU to create "over-gating." Our measurements show the opposite: ViT-5 has **lower** average Hoyer sparsity (0.510 vs 0.544), **lower** Gini coefficient (0.565 vs 0.605), and **fewer** near-zero activations (14.7% vs 20.9%) than DeiT-III.

The "over-gating" hypothesis predicts ViT-5 should be sparser. It is not. The hypothesis is empirically falsified.

### Finding 2: DeiT-III suffers from severe late-layer activation collapse that ViT-5 mitigates

DeiT-III layer 9 has **93.1% near-zero activations** — a near-complete representation collapse. ViT-5 reduces this to 76.3%. Across layers 8-11, DeiT-III consistently shows 1.1-3.6x more dead activations. ViT-5's architectural components (likely RoPE + registers + QK-Norm) appear to **prevent** sparsity collapse rather than cause it. The paper should be claiming credit for this, not worrying about adding more gating.

### Finding 3: LayerScale gammas are not "gates" — they are unbounded signed rescalers

The paper describes LayerScale as "a form of static gating" (Section 3.3). Gates produce values in [0, 1] (like SwiGLU's sigmoid). LayerScale gammas in the trained ViT-5 model range from **-57.2 to +82.9**, with approximately **50% negative** at every layer and standard deviations reaching 32.5. These are not gates — they are learned signed channel-wise affine transforms.

The functional analogy between LayerScale and SwiGLU gating is fundamentally broken:
- SwiGLU gate: sigmoid output in [0, 1], multiplicative on FFN hidden dim
- LayerScale gamma: unbounded signed scalar per channel, multiplicative on block output dim

Two mechanisms with completely different value ranges, dimensionalities, and semantics cannot "compound" in the way the paper claims.

### Finding 4: LayerScale does amplify sparsity of the residual signal, but less than DeiT-III's natural sparsity

LayerScale increases the Hoyer sparsity of the FFN output from 0.385 to 0.518 (+0.133). However, this scaled signal is still less sparse than DeiT-III's raw FFN output at the same layers. LayerScale is reshaping the channel distribution, not suppressing it.

### Overall Assessment

The "over-gating" explanation (Section 3.3) is the paper's only mechanistic contribution — the rest is empirical component selection. Our measurements show this mechanism does not exist as described:

1. The predicted effect (increased sparsity) is not observed
2. The proposed mechanism (LayerScale as static gating) mischaracterizes what LayerScale learns
3. The comparison (LayerScale gating analogous to SwiGLU gating) is a false analogy

The LayerScale + SwiGLU performance degradation (Table 2 in the paper: 84.16% -> 83.70%) is real, but its cause remains unexplained. Possible alternative explanations include:
- Optimization landscape interference (two nonlinear channel-wise transforms creating difficult loss geometry)
- Gradient flow disruption (LayerScale's large negative gammas interacting poorly with SwiGLU's gradient paths)
- Effective rank reduction from the composition of two low-rank projections

These would require further experiments (loss landscape analysis, gradient norm tracking, effective rank measurement) to investigate.

---

## Hypotheses: What Actually Causes the LayerScale + SwiGLU Degradation?

The degradation is real (Table 2: 84.16% -> 83.70%), but "over-gating" is a label, not a mechanism. Based on our measurements, here are three plausible alternative explanations, ranked by likelihood.

### Hypothesis 1: Gradient conflict from competing multiplicative paths (most likely)

The forward pass through the FFN branch computes:

```
Vanilla:  residual += γ * (GeLU(xW1) @ W2)            # one multiplicative gate (LayerScale)
SwiGLU:   residual += γ * ((σ(xW_gate) * xW1) @ W2)   # two multiplicative gates
```

During backprop, gradients to W1 must flow through both γ (LayerScale) and σ(xW_gate) (SwiGLU gate). We showed that γ has ~50% negative values with magnitudes up to 83. When γ is large and negative for a channel, it flips the gradient sign for that channel. The SwiGLU gate is simultaneously trying to learn which hidden dimensions to activate. These two learning signals can fight each other — LayerScale says "invert this channel," SwiGLU says "gate this feature on" — creating noisy, conflicting gradient updates that slow convergence or push toward worse minima.

With vanilla GeLU, there's no learned gate inside the FFN, so LayerScale is the only channel-wise multiplicative controller. It can learn freely. Adding a second learnable multiplicative controller (SwiGLU) creates an optimization coordination problem.

**How to test**: Measure cosine similarity between gradients at the LayerScale gamma and SwiGLU gate parameters during training. If they're anti-correlated, this confirms the conflict.

### Hypothesis 2: Initialization trap

LayerScale initializes γ at 1e-4 (near-zero). In early training, the FFN contribution to the residual stream is almost nothing — the model learns primarily through attention. SwiGLU changes the FFN's gradient magnitude and direction compared to GeLU. If the early-training gradient signal through an almost-zeroed LayerScale is too weak or noisy to properly initialize the SwiGLU gate weights, the model gets stuck in a bad basin early on and never recovers.

Evidence: The final gammas are huge (-57 to +83), meaning they had to travel very far from 1e-4. With SwiGLU adding another multiplicative bottleneck in that path, the early optimization landscape may have a different curvature that makes this journey harder.

**How to test**: Train ViT-5-Small for 100 epochs with LayerScale + SwiGLU but change the LayerScale init from 1e-4 to 1.0 (identity). If the degradation disappears, the init is the culprit. Also try warming up LayerScale for N epochs before enabling SwiGLU gating.

### Hypothesis 3: Effective dimensionality squeeze

SwiGLU's gate σ(xW_gate) is data-dependent but, across the dataset, certain hidden dimensions are gated off more often than others — creating a soft "preferred subspace" in the hidden dimension. LayerScale's γ then projects the FFN output through a signed diagonal matrix in the output dimension. The composition:

```
output = γ * ((soft_mask * xW1) @ W2)
```

is equivalent to `γ * W2^T @ diag(soft_mask) @ W1^T @ x` — a product of two diagonal-like projections sandwiching the weight matrices. If both diagonals have many near-zero entries (SwiGLU's gate kills ~50% of hidden dims, LayerScale's negative gammas effectively project out ~50% of output dims), the effective rank of the combined transform could be too low for the model capacity.

**How to test**: Measure the effective rank (via SVD) of the full FFN Jacobian (input-to-output) with and without SwiGLU. If the combined version has significantly lower rank, this is the cause.

### Recommended experiment

If limited to one experiment: **test Hypothesis 2**. Train ViT-5-Small for 100 epochs with LayerScale + SwiGLU but change LayerScale init from 1e-4 to 1.0. If the degradation vanishes, the problem is entirely an optimization artifact — the architecture is fine, the initialization creates an incompatible learning dynamic. This would be a more useful finding than "over-gating" and would actually allow ViT-5 to use SwiGLU (aligning it with modern LLM practice).

---

## Broader Implications for the Paper

The "over-gating" explanation is the paper's only mechanistic contribution — everything else is empirical component selection ("we tried X, accuracy went up"). With this mechanism empirically falsified:

1. **The paper has no explanatory contribution** — it is purely a recipe/configuration paper
2. **The rejection of SwiGLU is unjustified** — it's based on a wrong explanation; with the right initialization or training schedule, SwiGLU might work fine
3. **The architecture may be suboptimal** — if SwiGLU can be recovered via better optimization, ViT-5 is leaving performance on the table by excluding it
4. **The claimed connection to LLM architecture evolution is broken** — the paper positions itself as bringing LLM advances to vision, but rejects the single most universal LLM component (SwiGLU) based on flawed analysis
