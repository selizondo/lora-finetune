# ADR-01: QLoRA vs Full Fine-Tuning

**Status:** Accepted  
**Date:** 2026-05-10  
**Applies to:** `lora-finetune/` and `finetune-case-study/`

---

## Context

Fine-tuning a 7B-parameter model on domain Q&A data requires choosing between full fine-tuning (all parameters updated) and parameter-efficient methods. The decision drives GPU requirements, training cost, deployment artifact size, and practical accessibility.

Full Mistral-7B fine-tuning in fp16 requires ~56GB VRAM for weights + optimizer states + gradients — beyond a single consumer GPU and outside the free-tier Colab allocation (A100: 40GB).

---

## Decision

Use QLoRA: 4-bit base weights (`bitsandbytes` nf4 quantization) + LoRA adapters trained on ~0.5% of parameters.

Concretely:
- Base model loaded in 4-bit: ~4GB VRAM
- LoRA adapters (r=16, α=32): ~8M trainable parameters vs 7B total
- Total VRAM at batch_size=4: ~14–16GB — fits on an A100 free tier or RTX 3090/4090

---

## Tradeoffs

**QLoRA wins:**
- Fits on a single 16GB GPU or free Colab A100
- Deployment artifact is the adapter file (~64MB) rather than full weights (~14GB)
- Training time: ~40 min on A100 for 1000 examples vs ~3–4h for full fine-tune
- For domain adaptation with <10K examples, QLoRA empirically matches full fine-tune quality within noise (the LoRA hypothesis: domain adaptation lives in a low-dimensional subspace)

**Full fine-tuning wins:**
- No quantization noise in base weights
- Adapts all parameters — better for tasks requiring deep architectural changes (not just domain vocabulary)
- No adapter loading overhead at inference
- Required when the base model's weights need significant restructuring (language transfer, task format changes)

**Chosen:** QLoRA. The task is domain adaptation (ML interview Q&A), not architectural change. <10K examples. Single-GPU constraint. Quality difference is within noise at this data scale.

---

## Boundary: When to Revisit

Switch to full fine-tuning when:
- Training data exceeds ~50K examples (the quality gap widens with scale)
- The task requires deep format changes that adapters can't express
- Inference infrastructure can absorb 14GB model weight storage per variant
- Quantization noise is measurably hurting output quality (test with `evaluate.py --mode both`)

---

## Consequences

- Perplexity slightly higher than full fine-tune on the same data (quantization noise)
- Adapter files must be loaded on top of the correct base model version — adapter ↔ base model version is a hard dependency
- Inference requires loading both base (4-bit) and adapter — two-step initialization vs one-step for full weights
- Flash Attention 2 is compatible with QLoRA on Ampere+ GPUs but omitted here for T4 compatibility; adds ~30% speed and reduces peak VRAM by ~20%
