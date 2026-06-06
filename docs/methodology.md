# Evaluation Methodology

How the rank sweep and data scaling experiments are designed, what each metric measures, and where the methodology has known limitations.

---

## Experiment Design

### Two experiments, one baseline

**Rank sweep:** Fixed 1000 training examples, 3 epochs. Vary rank: r=8, r=16, r=64. All other hyperparameters held constant (`config.yaml`). Measures: adapter quality as a function of capacity.

**Data scaling:** Fixed rank r=16, 3 epochs. Vary examples: n=100, n=500, n=1000. Measures: quality as a function of training data volume.

**Baseline:** Raw Mistral-7B-v0.1 with no adapter. All fine-tuned results reported relative to this baseline.

### Held-out validation set

The validation set is split before any training. MD5(question) deduplication ensures no training question appears in the validation set. Val set size: ~600 examples (10% of ~6000 usable after quality filters). The fine-tuned model's perplexity is measured on val.jsonl.

### BM25 comparison baseline (no GPU)

`baseline/bm25_baseline.py` implements a retrieval-based comparison: for each test question, retrieve the top-1 training answer by BM25 similarity and report ROUGE-L and exact-match. This provides a no-GPU baseline to compare against the fine-tuned model. If the fine-tuned model scores lower than BM25 on this benchmark, fine-tuning did not help.

---

## Metrics

### Perplexity

Primary metric for the rank sweep and data scaling experiments. Perplexity on the validation set measures how well the model predicts held-out answers. Lower is better. Perplexity is sensitive to temperature and tokenization, so all comparisons in this repo use the same model family (Mistral-7B), the same tokenizer, and the same sequence length (seq_len=256 on T4).

**Limitation:** Perplexity measures prediction quality, not answer quality. A model can have low perplexity (predicts validation tokens well) while producing factually incorrect answers. Task-specific metrics (accuracy, hallucination rate) are better but require an LLM judge or human annotation. See docs/evidence.md for why the two approaches disagree at small data scales.

### LLM-as-judge qualitative comparison

`evaluate.py --mode qualitative` runs a side-by-side comparison: the base model and the fine-tuned model both answer 20 held-out questions, scored by the LLM judge on correctness (1 to 3) and hallucination (bool). This produces relative scores for the same 20 questions, making the comparison meaningful even with a small sample.

**Limitation:** The judge scores are not calibrated against human annotation. They are useful for direction (did fine-tuning help or hurt?) but not for absolute accuracy claims.

### ROUGE-L and exact-match (BM25 baseline only)

Used for the BM25 baseline comparison because ROUGE is interpretable for retrieval-based results: a retrieved verbatim answer from the training set would score near 1.0 on ROUGE-L. The fine-tuned model's ROUGE-L will be lower because it generates rather than retrieves, which is expected and acceptable as long as the generated answer is correct.

---

## Data Preparation

### Source

`win-wang/Machine_Learning_QA_Collection` from HuggingFace (~8600 Gemma-format conversations).

### Quality filters

Applied in `prepare_dataset.py`:
- Skip questions under 20 characters (noise, fragments)
- Skip answers under 30 characters (non-answers, "I don't know" responses)
- Result: ~8000 usable examples

### Format conversion

Gemma chat format (`<start_of_turn>user/model`) → Alpaca-style JSONL → Llama-2 instruction template (`[INST] <<SYS>> ... <</SYS>>`). The template conversion is a silent correctness requirement: the fine-tuned model must see the same template at inference as at training. Mismatch degrades instruction-following without an error.

---

## Known Limitations

**Perplexity is not accuracy.** The rank sweep shows perplexity converges quickly with rank, but a separate LLM-as-judge run is needed to verify that perplexity improvements translate to answer quality improvements. At small data scales, they may not.

**Single base model.** All experiments use Mistral-7B-v0.1. A stronger instruction-tuned base (Llama-3-8B-Instruct) would show different breakeven points for rank and data volume. The results here are not universal: they are specific to this base model and this task.

**No semantic deduplication.** Near-identical questions with different wording are not caught by MD5. Inflated fine-tune scores are possible if training-similar questions appear in the validation set.

**T4 seq_len constraint.** T4 runs use seq_len=256 due to VRAM constraints. Longer sequences (512) would capture more context but require a larger GPU. All perplexity results are seq_len=256.
