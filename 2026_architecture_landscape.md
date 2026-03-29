# 2026 LLM / VLM Architecture Landscape

What's actually standard vs what ViT-5 does.

---

## The 2026 LLM Default Recipe

Every major model family (LLaMA 3/4, Qwen3, Gemma 3, DeepSeek-V3, Mistral 3) has converged on:

| Component | Standard | Details |
|-----------|----------|---------|
| Normalization | Pre-RMSNorm | Universal. Post-norm and LayerNorm are dead. Gemma uses sandwich norm (pre+post) for extra stability. |
| FFN | SwiGLU | Universal. Expansion ratio ~8/3 * d_model (~3.5x). No major model uses plain GeLU anymore. |
| Positional encoding | RoPE | Universal. theta=500K (LLaMA 3), 1M (Qwen2.5). ALiBi and learned APE are dead. |
| Attention | GQA | 8 KV heads for most sizes. MQA is dead. Full MHA only for <3B models. DeepSeek uses MLA (compressed latent KV). |
| QK-Norm | Trending standard | DeepSeek-V3, Gemma 3 (via soft-capping), Cohere. Not yet universal but growing. |
| Biases | None anywhere | QKV, FFN, output projections — all bias-free. Qwen is the exception (bias on Q, K only). |
| Scale | MoE for >70B | Dense models above 70B are rare. Top-2 routing, 8+ experts. |

### What's been abandoned (was used 2023-2024, now dead)

- **Post-LayerNorm** — replaced by Pre-RMSNorm
- **Plain GeLU/ReLU FFN** — replaced by SwiGLU
- **ALiBi** — replaced by RoPE
- **Learned absolute positional embeddings** — replaced by RoPE
- **Multi-Query Attention** — replaced by GQA
- **Full MHA at >7B scale** — replaced by GQA

---

## The 2026 VLM Default Recipe

### Vision Encoder

| VLM | Vision Encoder | Key Detail |
|-----|---------------|------------|
| Qwen2.5-VL / Qwen3-VL | Custom ViT 675M with **2D-RoPE** | Native dynamic resolution, no APE |
| InternVL 2.5/3 | InternViT-6B | Massive 6B-param ViT, tile-based dynamic res |
| LLaVA-OneVision | **SigLIP SO400M/14@384** | The "default" open-source encoder |
| Cambrian-1 | SigLIP + DINOv2 ensemble | Dual encoder: semantic + spatial |
| Prismatic | SigLIP + DINOv2 | Confirmed dual > single |
| Phi-3.5-Vision | Custom ViT 400M | Dynamic tile cropping |
| Gemini 2.0 / GPT-4o | Natively multimodal | No separate encoder |

**The clear trend**: SigLIP has completely replaced CLIP as the default encoder. SigLIP-2 adds NaFlex (native flexible resolution).

### Resolution Handling

| Era | Approach |
|-----|----------|
| 2023 | Fixed 224x224 or 336x336 |
| 2024 | AnyRes tiling (split into fixed-size tiles) |
| 2025-2026 | **Native dynamic resolution** via 2D-RoPE or masked attention |

Fixed-resolution encoding is dead. Tiling is the current workhorse but has known issues (boundary artifacts, token explosion). Native resolution (Qwen2.5-VL, SigLIP-2 NaFlex) is the frontier.

### Connector (Vision → LLM)

| Era | Approach |
|-----|----------|
| 2023 | Linear projection (LLaVA v1) — dead |
| 2023 | Q-Former / Perceiver Resampler (BLIP-2, Flamingo) — dead |
| 2025-2026 | **2-layer MLP + spatial token merging** (standard) |

Q-Former and Perceiver Resampler are dead — they over-compress spatial information. Simple MLP + spatial merge (e.g., 2x2 tokens → 1) is preferred.

### What's been abandoned in VLMs

- **CLIP ViT-L/14** as default encoder — replaced by SigLIP
- **Fixed resolution processing** — replaced by dynamic/native
- **Q-Former / Perceiver Resampler** — replaced by simple MLP connector
- **Cross-attention connector** — simple MLP preferred

---

## Diffusion / Generation Architectures

| Component | Standard |
|-----------|----------|
| Architecture | **DiT** (Diffusion Transformer). U-Net is dead for frontier models. |
| Conditioning | **AdaLN-Zero** (not cross-attention) |
| Text-image interaction | **MM-DiT joint attention** (text and image tokens in same attention) |
| Positional encoding | **RoPE / 2D-3D RoPE** (replacing absolute PE) |
| Training objective | **Flow matching** (replacing DDPM) |
| QK-Norm | Essential for DiT training stability |

---

## Emerging (Not Yet Standard)

| Trend | Status |
|-------|--------|
| Unified vision-language transformers | Gemini/GPT-4o do this, open models don't yet |
| MoE in vision encoders | Explored (V-MoE) but not adopted in production VLMs |
| Mamba/SSM for vision | Published (Vim, VMamba) but zero adoption in real VLMs |
| Byte-level tokenization | Experimental, BPE still dominant |
| Linear attention | Still experimental, softmax + FlashAttention is standard |

---

## How ViT-5 Compares to 2026 Standards

| Component | 2026 Standard | ViT-5 Choice | Assessment |
|-----------|---------------|-------------|------------|
| Normalization | Pre-RMSNorm | Pre-RMSNorm | Aligned |
| FFN | SwiGLU | **GeLU (rejects SwiGLU)** | **Against standard** — based on flawed "over-gating" analysis |
| Positional encoding | RoPE only (high theta) | APE + RoPE | **Partially aligned** — keeping APE is non-standard and causes over-smoothing |
| QK-Norm | Increasingly standard | Yes | Aligned |
| Biases | None | None | Aligned |
| Attention structure | GQA | Full MHA | Acceptable for Small/Base (GQA matters more at scale) |
| Resolution | Native dynamic | Fixed 224 | **Behind** — no dynamic resolution support |
| Registers | Not standard anywhere | 4 registers + RoPE hack | **Non-standard** — fragile, unique to this paper |
| LayerScale | Not used in LLMs | Yes (init 1e-4) | **Non-standard** — LLMs use post-norm for same effect |

### The gap

ViT-5 claims to bring "LLM advances to vision" but:
1. **Rejects the most universal LLM component** (SwiGLU) based on wrong analysis
2. **Keeps the most obsolete component** (absolute positional embeddings) that LLMs abandoned in 2023
3. **Adds components no LLM uses** (registers, LayerScale)
4. **Ignores the biggest VLM trend** (dynamic resolution)

The actual "LLM recipe for vision" in 2026 is: **RMSNorm + SwiGLU + 2D-RoPE + QK-Norm + no biases + dynamic resolution**. Qwen2.5-VL already does exactly this. ViT-5 gets 3 of 6 right.
