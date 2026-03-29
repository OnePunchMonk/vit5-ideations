# Experiment 3: Token Diversity & Attention Entropy — Results

## Objective

Test whether ViT-5's components (RoPE, registers, QK-Norm) actually improve spatial modeling and prevent over-smoothing, as claimed in the paper. The paper only shows qualitative attention visualizations (Figure 4) — we quantify it.

## Setup

- **Models**: ViT-5-Small (22M, pretrained) vs DeiT-III-Small (22.1M, pretrained)
- **Data**: CIFAR-100 test set, 200 images, 224x224, batch_size=16
- **Metrics**: Token cosine similarity (over-smoothing), attention entropy (diffuseness), cross-head similarity (head redundancy), CLS attention entropy (classification focus)
- **Hardware**: T4 16GB (Colab free tier)

---

## Raw Data

### Per-Layer Token Diversity & Attention Entropy

```
Layer | TokenSim(V5) TokenSim(D3)    diff | AttnEnt(V5) AttnEnt(D3)    diff | HeadSim(V5) HeadSim(D3)
------+--------------+------------+-------+-------------+------------+------+-------------+-----------
    0 |       0.6637       0.2925 +0.3712 |      4.1614      3.3133 +0.8481 |      0.4343      0.6312
    1 |       0.6403       0.3736 +0.2667 |      3.9619      3.8590 +0.1029 |      0.4413      0.5694
    2 |       0.6398       0.4097 +0.2301 |      2.4794      2.6716 -0.1922 |      0.2220      0.2820
    3 |       0.6245       0.4220 +0.2025 |      3.1124      2.4163 +0.6960 |      0.2857      0.2186
    4 |       0.5728       0.4182 +0.1546 |      3.6165      2.9958 +0.6206 |      0.2923      0.3118
    5 |       0.4948       0.3933 +0.1015 |      3.1443      3.6150 -0.4707 |      0.2990      0.4273
    6 |       0.3984       0.3461 +0.0523 |      3.2741      3.1448 +0.1293 |      0.4840      0.4873
    7 |       0.3761       0.3577 +0.0184 |      3.1384      3.1086 +0.0298 |      0.2783      0.7006
    8 |       0.3508       0.3868 -0.0359 |      3.0116      2.8167 +0.1949 |      0.3657      0.7254
    9 |       0.3882       0.4137 -0.0255 |      2.5514      2.5669 -0.0155 |      0.5909      0.8262
   10 |       0.3347       0.3055 +0.0292 |      2.5360      3.3914 -0.8554 |      0.7685      0.8418
   11 |       0.3930       0.4714 -0.0784 |      4.9928      3.9792 +1.0137 |      0.7527      0.8137
------+--------------+------------+-------+-------------+------------+------+-------------+-----------
  AVG |       0.4898       0.3825 +0.1072 |      3.3317      3.1566 +0.1751 |      0.4346      0.5696
```

### Summary Statistics

| Metric | ViT-5-Small | DeiT-III-Small | Delta | Interpretation |
|--------|-------------|----------------|-------|----------------|
| Avg token cos-sim | 0.4898 | 0.3825 | +0.1072 | ViT-5 MORE over-smoothed |
| Avg attention entropy | 3.3317 | 3.1566 | +0.1751 | ViT-5 slightly more diffuse |
| Avg cross-head sim | 0.4346 | 0.5696 | -0.1350 | ViT-5 heads MORE diverse |

### CLS Token Attention Entropy

```
Layer | CLS_Ent(V5) CLS_Ent(D3)    diff
------+-------------+-----------+-------
    0 |      2.9081      3.7091 -0.8010
    1 |      2.7699      2.7739 -0.0041
    2 |      0.6142      0.5537 +0.0605
    3 |      1.4448      1.1704 +0.2744
    4 |      1.4646      1.9380 -0.4734
    5 |      0.9777      1.9859 -1.0081
    6 |      1.3113      2.0455 -0.7342
    7 |      1.8010      1.2932 +0.5078
    8 |      1.9132      2.1048 -0.1916
    9 |      2.1563      2.0702 +0.0860
   10 |      2.5983      3.6690 -1.0706
   11 |      2.9936      3.8518 -0.8582
```

Avg CLS entropy: ViT-5 = 1.913, DeiT-III = 2.346. ViT-5's CLS token has **lower entropy (more focused attention)** in 8 out of 12 layers.

---

## Figures

- `exp3_figures/vit-fig1-2.png` — 3-panel comparison: token similarity, attention entropy, head redundancy
- `exp3_figures/vit-fig2-2.png` — Per-head attention entropy heatmaps (6 heads x 12 layers)
- `exp3_figures/vit-fig3-2.png` — CLS token attention entropy across layers

---

## Analysis

### Finding 1: ViT-5 is MORE over-smoothed than DeiT-III, not less

This is the most surprising result. ViT-5 has **higher** average token cosine similarity (0.49 vs 0.38) — meaning tokens are more similar to each other, which is the classic sign of over-smoothing.

The pattern is layer-dependent:
- **Layers 0-7**: ViT-5 is consistently more over-smoothed (+0.02 to +0.37 cos-sim)
- **Layers 8-9**: ViT-5 slightly less over-smoothed (-0.03 to -0.04)
- **Layer 11**: ViT-5 less over-smoothed (-0.08)

The early-layer over-smoothing is dramatic — layer 0 shows cos-sim of 0.66 (ViT-5) vs 0.29 (DeiT-III). This suggests that ViT-5's components (likely APE+RoPE dual positional encoding) cause early tokens to be more correlated from the start, and the model only differentiates them in later layers.

This contradicts the paper's narrative that ViT-5 has "improved spatial reasoning" — if tokens are more similar, the model has less spatial discrimination to work with.

### Finding 2: ViT-5 attention heads are more diverse (genuinely positive)

Cross-head similarity is lower in ViT-5 (0.43 vs 0.57), meaning different heads learn more distinct attention patterns. This is the one metric where ViT-5 clearly outperforms DeiT-III and it's a meaningful improvement — more diverse heads means the multi-head attention is being used more efficiently.

This likely comes from the combination of RoPE (which induces frequency-dependent positional patterns) and QK-Norm (which normalizes the attention logit scale, preventing heads from collapsing to similar patterns).

The improvement is most dramatic in mid-to-late layers:
- Layer 7: 0.28 vs 0.70 (ViT-5 heads 2.5x more diverse)
- Layer 8: 0.37 vs 0.73 (2x more diverse)
- Layer 9: 0.59 vs 0.83 (1.4x more diverse)

### Finding 3: ViT-5's CLS token has more focused attention

CLS entropy is lower in ViT-5 in 8/12 layers (avg 1.91 vs 2.35 nats). Lower entropy means the CLS token attends more selectively to specific patches rather than spreading attention uniformly. This is good for classification — it means the CLS token is extracting information from specific regions rather than averaging over everything.

The strongest focus differences appear at layers 5, 6, 10, and 11 (the layers closest to the classification head), where ViT-5's CLS entropy is 0.7-1.1 nats lower. This selective attention likely contributes to the +0.4% accuracy improvement.

### Finding 4: Attention entropy heatmap reveals head specialization

From the per-head heatmaps:
- **ViT-5**: Shows more variance across heads within each layer. Some heads have very low entropy (focused, attending to specific positions) while others have high entropy (global context). This specialization is healthy.
- **DeiT-III**: More uniform across heads, especially in late layers (9-11) where almost all heads have similar moderate entropy. This is the "attention collapse" phenomenon — heads become redundant.

ViT-5's layer 2 is notable: head 2 has very low entropy (~1.3 nats) while heads 4-5 have high entropy (~3.5 nats). This kind of within-layer diversity means the model is simultaneously capturing both local and global patterns.

---

## Conclusions

### The good news for ViT-5

1. **Head diversity is genuinely improved** — ViT-5 heads are 24% less redundant (0.43 vs 0.57 cross-head sim). This is a real architectural benefit, likely from RoPE + QK-Norm.
2. **CLS attention is more focused** — better for classification, the model knows where to look.
3. **Head specialization is stronger** — different heads serve different roles (local vs global).

### The bad news for ViT-5

1. **ViT-5 is MORE over-smoothed** — tokens are 28% more similar on average (0.49 vs 0.38 cos-sim). The paper claims "improved spatial reasoning" but the tokens are actually less differentiated.
2. **The over-smoothing is worst in early layers** — layer 0 cos-sim is 0.66 vs 0.29, a 2.3x difference. This suggests the dual positional encoding (APE+RoPE) may be causing tokens to start too correlated.
3. **The paper's attention visualizations (Figure 4) are misleading** — showing cleaner attention maps doesn't mean better spatial reasoning if the underlying tokens are more homogeneous. Clean attention over homogeneous tokens is not the same as discriminative attention over diverse tokens.

### Nuance: over-smoothing may not hurt classification

The +0.4% accuracy gain despite higher over-smoothing suggests that for ImageNet classification specifically, having more similar tokens plus focused CLS attention may work fine — the CLS token just needs to find the right region, not differentiate every patch. But for dense prediction tasks (detection, segmentation), over-smoothing is directly harmful because you need distinct per-token representations. This may explain why ViT-5's gains on ADE20K segmentation (+2.3 mIoU for Small, +2.7 for Large) come primarily from RoPE's spatial encoding rather than from improved token diversity.

---

## Broader Implications

The paper shows qualitative attention maps (Figure 4) as evidence of "improved spatial understanding." Our quantitative analysis reveals a more complex picture:

- ViT-5 has **better attention structure** (more diverse heads, more focused CLS)
- But **worse token diversity** (more over-smoothed representations)

These two effects partially cancel out. The net result is a model that is slightly better at classification (the CLS focus helps) but potentially weaker for tasks requiring fine-grained spatial discrimination (the over-smoothing hurts).

This reinforces the critique that ViT-5 is a recipe tuned for ImageNet, not a general-purpose backbone improvement. A truly improved spatial backbone should show BOTH better attention structure AND better token diversity.
