# Architectural Tradeoffs

Decisions made during build, with the reasoning and scale/complexity boundaries where each would need revisiting.

---

## QLoRA vs Full Fine-Tuning

**Decision:** QLoRA — 4-bit base weights (`bitsandbytes` nf4) + LoRA adapters trained on ~0.5% of parameters.

**Why:** Full Mistral-7B fine-tuning in fp16 requires ~56GB VRAM (weights + optimizer states + gradients) — beyond any single consumer GPU and the free-tier Colab A100 (40GB). QLoRA fits on a single 16GB GPU or free Colab A100:
- Base model in 4-bit: ~4GB VRAM
- LoRA adapters (r=16, α=32): ~8M trainable parameters vs 7B total
- Total at batch_size=4: ~14–16GB

**Tradeoff:** Full fine-tuning has no quantization noise and adapts all parameters. QLoRA has slightly higher perplexity (quantization noise) and creates a hard adapter ↔ base model version dependency — the adapter is only valid for the exact base model it was trained on.

**Scale boundary:** Switch to full fine-tuning when training data exceeds ~50K examples (quality gap widens with scale), when the task requires deep format changes that adapters can't express, or when inference infrastructure can absorb 14GB model weight storage per variant.

---

## LoRA Rank (r=16)

**Decision:** r=16 with α=32.

**Why:** The LoRA hypothesis is that domain adaptation lives in a low-dimensional subspace. For ML interview Q&A, r=16 captures the domain shift without overfitting to the small training set. r=4 underfits on specialized vocabulary; r=64 adds training time without measurable quality gain on <10K examples.

**Scale boundary:** Increase r when the task requires memorizing large amounts of factual content (not just style transfer). r=64–128 is common for tasks like code generation where broad factual coverage matters.

---

## Alpaca Prompt Format

**Decision:** `[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]` (Llama-2 instruction template).

**Why:** Mistral-7B-v0.1 was instruction-tuned with this template. Using a different format degrades instruction-following without raising an error — the model's behavior changes silently.

**Tradeoff:** Template is hardcoded in four files (`prepare_dataset.py`, `train.py`, `evaluate.py`, `inference.py`). Single source of truth for the template is the right fix (move to `config.yaml`); currently a documented gap.

---

## Sequential 80/10/10 Split

**Decision:** Temporal split (first 80% train, next 10% val, last 10% test) with MD5 deduplication on question text.

**Why:** Random splits on Q&A data risk the same question appearing in both train and test (memorization, not generalization). Temporal ordering reflects real usage where future questions build on earlier concepts.

**Boundary:** Assumes examples are in chronological or random-equivalent order; if the dataset is sorted by topic, the temporal split may under-represent some topics in training.

---

## LLM-as-Judge vs ROUGE Metrics

**Decision:** LLM judge (qwen2.5-coder:7b at temperature=0) scoring correctness, hallucination, and conciseness.

**Why:** ROUGE measures n-gram overlap, which correlates poorly with ML answer quality — a technically correct answer can use different wording than the reference and score near zero. LLM judges better reflect whether the answer actually addresses the question and avoids fabrication.

**Tradeoff:** Absolute scores are judge-model-specific. Results should be reported as relative comparisons across approaches (FT vs baseline vs RAG), not as ground-truth quality scores. A stronger judge (GPT-4o) would shift absolute numbers.

**Scale boundary:** Judge inference cost grows linearly with test set size. At 200+ test examples, batch evaluation on a cheaper model or embedding-based metrics become more practical.

---

## What Was Cut

| Cut | Reason | Upgrade trigger |
|---|---|---|
| Flash Attention 2 | T4 GPU incompatibility | Ampere+ GPU (A100, RTX 3090+) available |
| Full fine-tune as comparison | ~3–4h per run, cost-prohibitive for sweep | Dedicated GPU budget or managed training service |
| Multi-adapter merging | No use case in this eval | Multi-domain adaptation required |
