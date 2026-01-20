# **STAFF REVIEW: lora-finetune**

### **Executive Summary**

**Portfolio Signal: ✅ Excellent** | **Production Readiness: ⚠️ Gaps in versioning & observability**

This project demonstrates strong judgment about *when* fine-tuning is worth it (compares to RAG baseline, finds RAG wins), but lacks systems thinking about *where* fine-tuning breaks down (adapter ↔ base model version mismatch is a hard dependency, undocumented at runtime). The design is portfolio-quality; shipping requires wrapper code for version contracts, template validation, and instrumentation.

---

### **Architecture & Design**

**Stack**: QLoRA (Mistral-7B-v0.1, 4-bit quantization + LoRA adapters, r=16) on ML interview Q&A (~8K train examples). Runs on free Colab T4 (~80 min).

**What makes this Staff-quality:**

1. **Contract-first design** — `config.yaml` is single source of truth; all experiments use CLI overrides (`--rank`, `--examples`) rather than file edits
2. **Baseline always reported** — Every evaluation compares both base and fine-tuned perplexity; no claims without context
3. **Explicit scale boundaries** — [ADR-01](docs/adr-01-qlora-vs-full-finetune.md) documents when to switch strategies: >50K examples = consider full fine-tuning; <10K examples = QLoRA is sufficient
4. **Decision rationale in README** — Answers the hard question: for ML Q&A, fine-tuning achieves ~65% accuracy vs ~70% for RAG. Conclusion: **RAG wins** unless you have <1s latency requirement. This is *honest* experimentation, not marketing
5. **Reproducible evaluation** — Held-out validation set, consistent Alpaca formatting, fixed evaluation prompts

---

### **Production-Readiness Assessment**

#### ❌ **Critical Gap: No Adapter ↔ Base Model Version Checking**

The adapter depends on exact base model version (`mistralai/Mistral-7B-v0.1`), but there's no validation at load time.

**Silent Failure Scenario:**
```python
# inference.py loads adapter + base
# If user swaps to Mistral-7B-v0.2 in config:
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.2")  # ← different weights
model = PeftModel.from_pretrained(model, adapter_path)  # ← will load, but produces nonsense
```

**Fix:**
- Embed `base_model_id` in `adapter_config.json` during training
- Validate at inference time:
```python
adapter_manifest = json.load(open(f"{adapter_path}/adapter_config.json"))
if adapter_manifest["base_model_id"] != base_model:
    raise ValueError(f"Adapter built for {adapter_manifest['base_model_id']}, got {base_model}")
```

---

#### ❌ **Gap: Template Mismatch Detection**

[README](README.md#L102) documents this as "silent quality regression": Llama-2 instruction template is hardcoded but not validated.

**The Problem:**
- Mistral-7B-v0.1 expects: `<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]</s>`
- Using ChatML or raw text degrades instruction-following without raising an error
- Template appears in 4 files: `prepare_dataset.py`, `train.py`, `evaluate.py`, `inference.py`

**Staff-quality Fix:**
```python
# In config.yaml, add:
model:
  chat_template: "llama_2"  # moved from hardcoded strings

# In train.py + inference.py, validate:
from transformers import get_template_from_model_id  # or custom lookup
expected_template = get_template_from_model_id(cfg["model"]["chat_template"])
actual_tokenizer_template = tokenizer.chat_template
if actual_tokenizer_template and expected_template not in actual_tokenizer_template:
    raise ValueError(f"Template mismatch: expected {expected_template}, got {actual_tokenizer_template}")
```

---

#### ❌ **Gap: No GPU Memory or Resource Guards**

No checks before loading model. Batch size and seq_len are configured but not validated against available VRAM.

```python
# train.py should check:
available_memory = torch.cuda.get_device_properties(0).total_memory
required_memory = estimate_qlora_memory(
    model_params=7e9,
    batch_size=cfg["training"]["batch_size"],
    seq_len=cfg["model"]["max_seq_length"]
)
if required_memory > available_memory * 0.9:  # leave 10% headroom
    raise RuntimeError(f"Insufficient VRAM: need {required_memory/1e9:.1f}GB, have {available_memory/1e9:.1f}GB")
```

---

#### ⚠️ **Gap: Observability in Inference**

No version metadata in inference output. If something breaks, ops can't trace which adapter+base+commit caused it.

**Current:**
```bash
$ python inference.py --adapter ./checkpoints/final --prompt "What is LoRA?"
A: LoRA is a parameter-efficient technique...
```

**Production-ready:**
```python
response = {
    "answer": "LoRA is a parameter-efficient technique...",
    "metadata": {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "adapter_version": "abc1234",
        "inference_time_ms": 1234,
        "tokens_generated": 42,
        "temperature": 0.7,
    }
}
```

---

### **Evaluation Gaps**

| Metric | Current | Issue |
|--------|---------|-------|
| **Perplexity** | ✅ Base vs FT reported | Proxy metric; tells you loss but not task performance |
| **Accuracy** | 📝 Documented (~65% vs 70% RAG) | Not measured in code; can't verify from repo alone |
| **ROUGE/BERTScore** | ❌ Missing | Task-specific metric; critical for Q&A |
| **Instruction adherence** | ❌ Missing | Can't measure if model respects "be concise" system prompt |
| **Factuality** | ❌ Missing | For ML Q&A, fact grounding matters; RAGAS-style eval would help |

**Recommendation:**
```python
# In evaluate.py, add:
from datasets import load_dataset
from rouge_score import rouge_scorer

BENCHMARK_QA = [  # canonical Q&A pairs with ground truth
    ("What is LoRA?", "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that trains only ~0.5% of parameters."),
    ("Difference between RAG and fine-tuning?", "RAG retrieves and augments at inference; fine-tuning adapts the model weights. RAG is faster to update; fine-tuning is lower latency."),
    # ... 10-20 more
]

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = []
for q, ground_truth in BENCHMARK_QA:
    predicted = generate(model, tokenizer, q, cfg)
    score = scorer.score(ground_truth, predicted)
    scores.append(score)
avg_rouge_l = sum(s['rougeL'].fmeasure for s in scores) / len(scores)
print(f"Avg ROUGE-L: {avg_rouge_l:.3f}")
```

---

### **Data Quality Lens**

| Check | Status | Finding |
|-------|--------|---------|
| **Train/val split hygiene** | ✅ Documented | 90/10 split, but no verification of contamination |
| **Label leakage** | ⚠️ Assumed clean | No check that val examples don't appear in training set |
| **Source quality** | ❌ Unknown | Ground truth origin not documented (Gemma-generated? Hand-authored? Quality unknown) |
| **Deduplication** | ❌ Missing | Only length filters; no semantic deduplication of near-duplicate questions |
| **Metric rationale** | ✅ Good | Perplexity → task-specific metrics progression is correct |

**Recommendation:**
```python
# In prepare_dataset.py, add after loading dataset:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Detect near-duplicates in training set
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
train_questions = [ex["instruction"] for ex in train_examples]
tfidf = vectorizer.fit_transform(train_questions)
similarity = cosine_similarity(tfidf)

# Flag pairs with >0.9 similarity
duplicates = []
for i in range(len(similarity)):
    for j in range(i + 1, len(similarity)):
        if similarity[i][j] > 0.9:
            duplicates.append((i, j, similarity[i][j]))

print(f"Found {len(duplicates)} near-duplicate question pairs (>0.9 similarity)")
if duplicates:
    print("Examples:")
    for i, j, sim in duplicates[:3]:
        print(f"  {train_questions[i][:60]}... ↔ {train_questions[j][:60]}... ({sim:.2f})")
```

---

### **Failure Modes: Committed but Not Tested**

From [ADR-01](docs/adr-01-qlora-vs-full-finetune.md) and README, these are identified but **not caught by code**:

1. **Template mismatch** → "silent quality regression" (documented, no runtime check)
2. **Adapter ↔ base version mismatch** → nonsense output (documented, no validation)
3. **Quantization noise** → perplexity higher than full FT (acknowledged, no alerting)
4. **Model download timeout** → process hangs (not caught)
5. **GPU OOM** → process killed (not checked)

**Pattern**: These are *documented* in prose but *untested* in code. Production systems that depend on documented-but-untested assumptions fail first.

---

### **LLM/RAG-Specific Patterns**

**What's Excellent:**
- README makes an explicit **RAG vs FT tradeoff decision** with data: "~65% accuracy (FT) vs ~70% (RAG) for ML Q&A at this scale"
- This is rare and commendable — most projects just fine-tune without measuring alternatives

**What's Missing:**
- The RAG baseline isn't in the code — it's only in README
- No side-by-side perplexity comparison: FT + BM25 retrieval vs fine-tuned-only
- No faithfulness metric (fine-tuning might hallucinate; RAG is grounded)

**Recommendation:**
```python
# In evaluate.py, add:
def eval_with_retrieval_baseline():
    """Compare fine-tuned model vs retrieval-augmented baseline."""
    from rank_bm25 import BM25Okapi
    
    # Index training data
    train_questions = [ex["instruction"] for ex in load_train_data()]
    train_answers = [ex["output"] for ex in load_train_data()]
    bm25 = BM25Okapi([q.split() for q in train_questions])
    
    # Evaluate on benchmark
    total_rouge = {"ft": 0, "rag": 0}
    for q, ground_truth in BENCHMARK_QA:
        # Fine-tuned only
        ft_answer = generate(ft_model, tokenizer, q, cfg)
        ft_score = rouge_score(ground_truth, ft_answer)
        total_rouge["ft"] += ft_score
        
        # RAG baseline
        top_k_docs = bm25.get_top_n(q.split(), train_answers, n=3)
        augmented_prompt = f"{q}\n\nContext:\n" + "\n".join(top_k_docs)
        rag_answer = generate(base_model, tokenizer, augmented_prompt, cfg)
        rag_score = rouge_score(ground_truth, rag_answer)
        total_rouge["rag"] += rag_score
    
    print(f"Fine-tuned ROUGE-L: {total_rouge['ft'] / len(BENCHMARK_QA):.3f}")
    print(f"RAG ROUGE-L: {total_rouge['rag'] / len(BENCHMARK_QA):.3f}")
```

---

### **Anti-Patterns Flagged**

| Pattern | Where | Fix |
|---------|-------|-----|
| Magic numbers | `seq_len=256` vs `seq_len=512` across files | Document why; make conditional on GPU type |
| Template hardcoded | 4 files: prepare_dataset.py, train.py, evaluate.py, inference.py | Move to config.yaml; validate at runtime |
| Observability post-hoc | ADR-01 notes template mismatch but no runtime check | Add version guards at load time |
| No error handling | HF download, GPU OOM, missing weights | Wrap with try/except; emit readable errors |
| No tests | No unit tests, no CI/CD, no regression suite | Add Pytest + GitHub Actions |

---

### **Actionable Recommendations (Ranked by Impact)**

#### 🔴 **High (System Design)**

1. **Add version manifest** (~1-2 hours)
   - Embed `base_model_id` + `lora_rank` + `config_hash` in adapter_config.json
   - Validate at inference time:
     ```python
     adapter_manifest = json.load(open(f"{adapter_path}/adapter_config.json"))
     if adapter_manifest["base_model_id"] != base_model:
         raise ValueError(f"Adapter built for {adapter_manifest['base_model_id']}, got {base_model}")
     ```

2. **Template to config + validation** (~1 hour)
   - Move hardcoded Llama-2 template to `config.yaml` under `model.chat_template`
   - Add runtime check in train.py, evaluate.py, inference.py:
     ```python
     if tokenizer.chat_template and cfg["model"]["chat_template"] not in tokenizer.chat_template:
         raise ValueError("Template mismatch")
     ```

3. **Observability wrapper** (~1-2 hours)
   - Add `inference_metadata()` helper that returns dict with version, latency, token_count
   - Log to file or embed in response
   - Test: verify metadata changes when adapter changes

4. **Contract test suite** (~2 hours)
   - Test adapter loads only with correct base model (should fail with wrong base)
   - Test template mismatch is caught
   - Test GPU memory pre-flight check
   - Add to Pytest, integrate with GitHub Actions

#### 🟡 **Medium (Evaluation)**

5. **Task-specific metrics** (~3 hours)
   - Add ROUGE-L + exact-match accuracy on 20-30 curated Q&A pairs
   - Compare base vs fine-tuned; report both
   - Add to `evaluate.py --mode metrics`

6. **RAG baseline in code** (~2 hours)
   - Implement BM25 retrieval over training data
   - Compare FT-only vs RAG on same benchmark
   - Confirm README claim: RAG ~70%, FT ~65% (or update if wrong)

7. **Instruction adherence test** (~1.5 hours)
   - 5-10 prompts asking for conciseness, specific format, etc.
   - Measure output length, structure compliance
   - Example: "Answer in one sentence" → measure if output is ≤2 sentences

#### 🟢 **Low (Polish)**

8. **Data quality audit** (~1-2 hours)
   - Semantic deduplication: flag near-duplicate questions (>0.9 cosine similarity)
   - Train/val contamination check: exact + fuzzy match between splits
   - Sample 20 training examples and audit for hallucination/noise

9. **CI/CD** (~1-2 hours)
   - GitHub Actions: test adapter → base model loading
   - Validate config.yaml YAML schema
   - Run evaluate.py on sample dataset

10. **Error handling** (~1 hour)
    - Wrap HF model downloads with timeout + retry
    - GPU memory pre-flight check before loading
    - Missing config file → helpful error message

---

### **Comparison to Your Other Projects**

**vs `rag-eval-pipeline`:**
- ✅ More honest about baselines (RAG vs FT comparison)
- ❌ Less rigorous evaluation (no RAGAS, no faithfulness metrics)
- ❌ No version-scoped metadata filtering (but not applicable to FT)

**vs `staff-ai-realtime-recsys`:**
- ✅ Cleaner contract design (config.yaml SOT)
- ❌ No training/serving skew detection
- ❌ No retrieval cascade with failure detection

---

### **Summary**

**Portfolio Signal:** ✅ **Excellent**
- Clear narrative about when to fine-tune (data-driven decision)
- Design decisions are documented (ADR-01)
- Reproducible experiments (config-driven, CLI sweeps)
- Good README storytelling

**Production Signal:** ⚠️ **Gaps, but Fixable**
- Adapter ↔ base model versioning not enforced (hard dependency, undocumented at runtime)
- Template mismatch documented as silent failure, but no runtime check
- Evaluation is perplexity-only; task-specific metrics missing
- No instrumentation for ops/debugging
- All gaps are addressable in 10-15 hours of work