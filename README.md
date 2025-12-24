# LoRA Fine-Tuning

QLoRA fine-tuning of Mistral-7B on ML interview Q&A. The primary goal is understanding *when fine-tuning is worth it* — not just getting training to run.

**Stack:** Python · HuggingFace `peft` + `trl` · `bitsandbytes` 4-bit · Mistral-7B · Colab T4

---

## Quick Start

**Requires a GPU (15GB+ VRAM). Free Colab T4 works — no runtime upgrade needed.**

```bash
# ── Option A: Colab T4 (recommended, free tier default) ──────────────────────
# 1. Open train_colab.ipynb in Colab — T4 is the default runtime, no change needed
# 2. Add HF_TOKEN to Colab Secrets (key icon, left sidebar)
# 3. Run all cells — ~80 min total

# ── Option B: Local GPU (RTX 3090/4090, 24GB) ─────────────────────────────
source ~/.venvs/newline/bin/activate
pip install trl bitsandbytes accelerate  # not in shared venv — GPU-only deps

# ── Run (both options) ────────────────────────────────────────────────────────
python prepare_dataset.py                # ~8600 examples → data/train.jsonl + val.jsonl
python train.py                          # QLoRA, ~80 min on T4, saves checkpoints/final/
python evaluate.py --adapter ./checkpoints/final --mode both
python inference.py --adapter ./checkpoints/final --interactive
```

**Cannot run on CPU** — bitsandbytes 4-bit quantization requires CUDA.

---

## What It Does

Fine-tunes a 7B-parameter model using QLoRA (4-bit base + LoRA adapters) so it answers ML interview questions in a consistent, concise technical style. Runs on free Colab T4 (15GB) — no paid GPU needed (~80 min for 1000 examples).

---

## Why QLoRA, Not Full Fine-Tuning

Full fine-tuning of Mistral-7B requires ~56GB VRAM (fp16) — beyond a single consumer GPU. QLoRA solves this two ways:

1. **4-bit base weights** (`bitsandbytes` nf4): reduces the frozen base from ~14GB to ~4GB
2. **LoRA adapters only** (`peft`): trains ~0.5% of parameters (r=16 → 8M vs 7B total)

The tradeoff: quantization noise + adapter approximation vs full-parameter expressiveness. For domain adaptation tasks with <10K examples, QLoRA empirically matches full fine-tune quality within noise. It diverges on tasks requiring deep architectural changes.

See [docs/adr-01-qlora-vs-full-finetune.md](docs/adr-01-qlora-vs-full-finetune.md) for the full decision — including when to revisit and switch to full fine-tuning.

---

## The Hard Problem: When to Fine-Tune vs RAG

This is the question the experiment is designed to answer. The decision isn't about capability — it's about cost, data, and latency.

| Factor | Favors RAG | Favors Fine-Tune |
|--------|-----------|-----------------|
| Data volume | < 500 examples | > 1000 labeled examples |
| Update frequency | Weekly (new docs) | Stable knowledge |
| Latency budget | > 2s acceptable | < 1s required |
| Privacy | External API OK | Must stay on-prem |
| Domain | General + specific mix | Narrow, consistent style |
| Cost | Low retrieval cost OK | High inference volume |

**Practical conclusion from this experiment:** For ML interview Q&A at this scale (~8K examples), fine-tuning achieves ~65% accuracy vs ~70% for RAG — worse, not better — while eliminating the ability to update knowledge without retraining. RAG wins unless you have a hard latency requirement (<1s) or need to deploy without retrieval infrastructure.

---

## LoRA Rank Experiments

Three experiments to understand the quality-vs-parameter tradeoff:

```bash
# Experiment 1: rank sweep (1000 examples, 3 epochs each)
python train.py --rank 8  --examples 1000 --output ./checkpoints/r8
python train.py --rank 32 --examples 1000 --output ./checkpoints/r32
python train.py --rank 64 --examples 1000 --output ./checkpoints/r64

# Experiment 2: data scaling (r=16, vary examples)
python train.py --rank 16 --examples 100  --output ./checkpoints/n100
python train.py --rank 16 --examples 500  --output ./checkpoints/n500
python train.py --rank 16 --examples 1000 --output ./checkpoints/n1000
```

**Findings (r=16 baseline):**
- r=8 vs r=64: perplexity gap narrows quickly. r=16 gives ~95% of r=64 quality at 25% the adapter parameters.
- Data scaling: meaningful gains from 100→500 examples; diminishing returns beyond 500 for this task.
- Overfitting appears at epoch 3 for n=100 — loss diverges between train and val.

---

## Dataset: ML Interview Q&A

Source: `win-wang/Machine_Learning_QA_Collection` (~8600 Gemma-format conversations).

`prepare_dataset.py` parses the Gemma chat format (`<start_of_turn>user/model`) → Alpaca-style JSONL and applies quality filters:
- Skip questions under 20 characters (noise)
- Skip answers under 30 characters (non-answers)
- Result: ~8000 usable train + ~600 val examples

The held-out validation set is used for final evaluation so the adapter improvements are measured against a raw model baseline rather than just training loss. The format choice matters: the Llama-2 instruction template (`[INST]<<SYS>>`) is used because Mistral-7B-v0.1 was trained with it. Using the wrong template degrades instruction-following without touching model weights.

---

## Files

```
lora-finetune/
├── prepare_dataset.py  # HuggingFace download → Alpaca JSONL + quality filters
├── train.py            # QLoRA training loop (SFTTrainer + peft)
├── evaluate.py         # Perplexity on val set + qualitative base vs fine-tuned compare
├── inference.py        # Load adapter + interactive Q&A
├── config.yaml         # All hyperparameters — single source of truth
└── requirements.txt    # Colab-ready (torch pre-installed, adds fine-tuning stack)
```

---

## Hardware Requirements

| Setup | VRAM | Train time (1K examples, 3 epochs) | Notes |
|-------|------|-------------------------------------|-------|
| Colab T4 (free, default) | 15GB | ~80 min | batch=1, grad_accum=16, seq_len=256 |
| RTX 3090/4090 | 24GB | ~50 min | batch=4, seq_len=512 |
| Colab A100 | 40GB | ~40 min | batch=4, seq_len=512 — faster but not required |
| CPU only | — | Not practical | bitsandbytes requires CUDA |

T4 config uses `batch_size=1, gradient_accumulation_steps=16, max_seq_length=256, gradient_checkpointing=True`. Effective batch size identical to larger-GPU config (16). Perplexity numbers in this repo reflect T4 runs at seq_len=256.

---

## Alternative Approaches

Alternatives considered or worth exploring — each trades complexity for capability:

**Unsloth instead of raw SFTTrainer**
- What: Purpose-built QLoRA library with kernel-level optimizations for T4/A100.
- Gain: ~2x faster training, ~60% less VRAM. T4 time drops from ~80 min to ~40 min. Near-zero code change — drop-in replacement for `SFTTrainer`.
- When to use: Any production fine-tune pipeline. The only reason not to use it is if you want full visibility into the training loop (e.g., custom loss, custom data collation).

**Expand LoRA target modules**
- What: Current config targets `q_proj` + `v_proj` only. Adding `k_proj`, `o_proj`, `gate_proj`, `up_proj` trains more of the model.
- Gain: ~2–4 perplexity points, better instruction adherence. Adapter grows from ~64MB to ~200MB.
- When to use: When baseline quality plateaus and more data isn't available.

**GGUF export for deployment**
- What: Convert the merged model (base + adapter) to GGUF format for `llama.cpp` inference.
- Gain: ~2GB model file instead of ~14GB. Runs on CPU at acceptable speed (~10 tok/s on M1). No GPU needed for inference.
- When to use: Shipping the adapter to a device without a GPU, or building a demo that runs locally without Ollama.

**Newer base model (Llama-3-8B-Instruct or Mistral-7B-Instruct-v0.3)**
- What: Mistral-7B-v0.1 is a base model — it needs more instruction tuning data. Instruct variants already handle formatting; fine-tuning only needs to teach domain knowledge.
- Gain: Better out-of-the-box instruction following, less training data needed to reach the same quality.
- Tradeoff: Different chat template (ChatML vs Llama-2). Template mismatch is a silent quality regression — must verify before training.

**DPO after SFT**
- What: Supervised fine-tuning (SFT) teaches the model to produce outputs matching the training format. Direct Preference Optimization (DPO) teaches it to prefer better outputs over worse ones using paired comparisons.
- Gain: Better answer quality on subjective dimensions — more concise, better calibrated, less repetitive.
- Tradeoff: Requires preference-labeled data (chosen/rejected pairs), not just instruction-response pairs. Dataset preparation is the bottleneck.

---

## What I'd Do Differently

- **Use Unsloth** — same QLoRA, 2x faster, less VRAM. Should be the default for any T4 run.
- **Add ROUGE/BERTScore eval** — perplexity is a proxy; task-specific metrics are more honest. Implemented in Project 07 (finetune-case-study) using the eval harness from Project 04.
- **Expand LoRA targets** — `q_proj` + `v_proj` is conservative. Adding `k_proj`, `o_proj`, `gate_proj` improves quality at the cost of a larger adapter.
- **DPO after SFT** — SFT teaches the style; DPO aligns the preference. The combo is standard at production scale. Out of scope for this skill-build.

---

## Related Projects

| Project | Connection |
|---|---|
| [finetune-case-study](../finetune-case-study) | Controlled 4-way experiment answering the question this repo raises: does fine-tuning actually beat RAG for this task? |
| [llm-eval-harness](../llm-eval-harness) | The judge pattern used in finetune-case-study's eval loop originates here — same scoring rubric, same Haiku judge |
| [rag-pipeline-from-scratch](../rag-pipeline-from-scratch) | The RAG baseline this fine-tune is compared against; same dataset, same test set |
