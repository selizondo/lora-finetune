# Design and Tradeoffs

Decisions made during build, with the reasoning and scale or complexity boundary where each breaks down.

---

## ADR-01: QLoRA vs Full Fine-Tuning

**Decision:** 4-bit base weights (bitsandbytes nf4 quantization) plus LoRA adapters trained on ~0.5% of parameters (r=16, alpha=32).

**Why not full fine-tuning:** Full Mistral-7B fine-tuning in fp16 requires ~56GB VRAM for weights, optimizer states, and gradients. That exceeds any single consumer GPU and the free Colab A100 (40GB). QLoRA fits on a single 16GB GPU or free Colab T4 at batch_size=1 with gradient accumulation:
- Base model in 4-bit: ~4GB VRAM
- LoRA adapters (r=16, alpha=32): ~8M trainable parameters vs 7B total
- Total at batch_size=1 with gradient accumulation: ~15GB

**Why QLoRA works here:** The LoRA hypothesis is that domain adaptation lives in a low-dimensional subspace. For ML interview Q&A (style + domain vocabulary, not architectural change), adapting ~0.5% of parameters captures the domain shift. Empirically, QLoRA matches full fine-tune quality within noise for fewer than 10K examples. The gap widens at 50K+ examples or when the task requires deep format changes.

**Tradeoffs:**
- Quantization noise: slightly higher perplexity than full fine-tune on the same data
- Adapter dependency: the adapter is valid only for the exact base model version it was trained on. Loading on Mistral-7B-v0.2 instead of v0.1 loads silently and produces degraded output.
- Two-step initialization at inference: load base (4-bit) then load adapter. One-step for full weights.
- Flash Attention 2 is compatible with QLoRA on Ampere+ GPUs but omitted for T4 compatibility. Adds ~30% speed and reduces peak VRAM by ~20% on A100.

**Scale boundary:** Switch to full fine-tuning when training data exceeds ~50K examples (quality gap widens), when the task requires deep format changes adapters cannot express, or when inference infrastructure can absorb 14GB weight storage per model variant.

---

## LoRA Rank: r=16

**Decision:** r=16 with alpha=32. Targets `q_proj` and `v_proj`.

**Why r=16, not r=4 or r=64:**
- r=4: underfits on specialized ML vocabulary. Perplexity does not converge to the same level as r=16.
- r=64: training time increases, adapter grows from ~64MB to ~200MB, perplexity improvement is within noise for fewer than 10K examples.
- r=16: gives ~95% of r=64 quality at 25% the adapter parameters. The rank sweep confirms this for ML Q&A.

**Why `q_proj` and `v_proj` only:** Conservative target selection. Adding `k_proj`, `o_proj`, `gate_proj`, and `up_proj` improves quality by approximately 2 to 4 perplexity points and improves instruction adherence, at the cost of a larger adapter (~200MB). The upgrade trigger: when baseline quality plateaus and more data is not available.

**Scale boundary:** Increase rank when the task requires memorizing large amounts of factual content (not just style transfer). r=64 to 128 is common for tasks like code generation where broad factual coverage matters.

---

## Prompt Format: Llama-2 Instruction Template

**Decision:** `[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]`

**Why:** Mistral-7B-v0.1 was instruction-tuned with this template. Using a different format degrades instruction-following without raising an error. The model's behavior changes silently based on whether the chat template matches what it saw during its own instruction tuning.

**Documented gap:** The template is currently hardcoded in four files (`prepare_dataset.py`, `train.py`, `evaluate.py`, `inference.py`). The correct fix is to move the template string to `config.yaml` as the single source of truth. A copy-paste divergence between files produces silent quality regression.

---

## 80/10/10 Split with MD5 Deduplication

**Decision:** Sequential 80/10/10 split; MD5(question) deduplicates across splits.

**Why:** Random splits on Q&A data risk the same question appearing in both train and test (memorization, not generalization). Temporal ordering reflects real usage. MD5 deduplication prevents exact-duplicate questions from appearing in both train and test.

**Known gap:** MD5 deduplication catches exact matches, not semantic duplicates. Near-identical questions with different wording pass the filter. A semantic deduplication pass at cosine similarity threshold of 0.95 would be more robust.

---

## LLM-as-Judge Over ROUGE

**Decision:** LLM judge (qwen2.5-coder:7b at temperature=0) scoring correctness, hallucination, and conciseness for qualitative comparison. ROUGE-L used for the BM25 baseline comparison only.

**Why:** ROUGE measures n-gram overlap, which correlates poorly with ML answer quality. A technically correct answer using different vocabulary from the reference scores near zero on ROUGE. LLM judges better reflect whether the answer addresses the question and avoids fabrication.

**Tradeoff:** Absolute scores are judge-model-specific. Report as relative comparisons (fine-tuned vs baseline on the same judge), not as ground-truth quality. A stronger judge would shift absolute numbers.

**Scale boundary:** Judge inference cost grows linearly with test set size. At 200+ test examples, batch evaluation on a cheaper model or embedding-based metrics become more practical.

---

## What Was Cut

| Cut | Reason | Upgrade trigger |
|-----|--------|-----------------|
| Flash Attention 2 | T4 GPU incompatibility | Ampere+ GPU (A100, RTX 3090+) |
| Full fine-tune comparison | ~3 to 4 hours per run | Dedicated GPU budget or managed training service |
| Expanded LoRA target modules | Adapter size and complexity | Quality plateau on current targets |
| GGUF export for deployment | Out of scope for skill-build | Shipping adapter to CPU-only device |
| DPO after SFT | Requires preference-labeled pairs | Alignment beyond instruction following |
| Unsloth as training backend | Added dependency for demo | Any production fine-tune pipeline |
