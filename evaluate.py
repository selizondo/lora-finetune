"""
evaluate.py — Evaluate fine-tuned model quality.

Two modes:
  1. Perplexity: compute perplexity on validation set (lower = better fit)
  2. Qualitative: side-by-side base vs fine-tuned answers on fixed test prompts

Usage:
    python evaluate.py --adapter ./checkpoints/final --mode perplexity
    python evaluate.py --adapter ./checkpoints/final --mode qualitative
    python evaluate.py --adapter ./checkpoints/final --mode both
"""

import argparse
import json
import logging
import math
import os
import time
from typing import List, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import (
    check_gpu_memory,
    validate_adapter_base_model,
    validate_chat_template,
    validate_tokenizer_template,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# System prompt: must match the one used in train.py/inference.py so the model
# sees the same instruction framing during eval that it was trained on.
SYSTEM_PROMPT = (
    "You are an expert ML engineer. Answer the following question clearly and concisely, "
    "as you would in a technical interview."
)

EVAL_PROMPTS = [
    "What is LoRA and why is it more efficient than full fine-tuning?",
    "What is the difference between RAG and fine-tuning?",
    "What is feature engineering and why does it matter?",
    "How do you handle class imbalance in a dataset?",
    "What is the vanishing gradient problem?",
]

# ---------------------------------------------------------------------------
# Item 4: Curated Q&A benchmark for task-specific metrics
# ---------------------------------------------------------------------------
# WHY a curated benchmark instead of the full val set: perplexity measures
# language-model fit, not whether answers are actually correct. ROUGE-L and
# exact-match on gold-standard Q&A pairs measure what an interviewer would
# actually care about: does the model produce the right answer?
# These 20 pairs are written against the ML interview domain the model was
# fine-tuned on. Ground-truth answers are concise canonical references —
# ROUGE-L measures n-gram overlap, exact-match catches verbatim recall.
BENCHMARK_QA: List[Tuple[str, str]] = [
    (
        "What is LoRA?",
        "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that trains "
        "only low-rank adapter matrices (~0.5% of parameters) instead of all model weights.",
    ),
    (
        "What is the difference between RAG and fine-tuning?",
        "RAG retrieves relevant documents at inference time and augments the prompt; "
        "fine-tuning adapts the model weights offline. RAG is faster to update; "
        "fine-tuning offers lower inference latency.",
    ),
    (
        "What is QLoRA?",
        "QLoRA combines 4-bit quantization of the base model with LoRA adapters, "
        "reducing VRAM requirements to ~4 GB for a 7B model while maintaining "
        "near-full fine-tune quality.",
    ),
    (
        "What is the vanishing gradient problem?",
        "Gradients shrink exponentially as they backpropagate through deep networks, "
        "making early layers train very slowly. Fixed by residual connections, "
        "batch normalization, and activation functions like ReLU.",
    ),
    (
        "What is feature engineering?",
        "The process of transforming raw data into features that improve model performance: "
        "normalization, encoding categoricals, creating interaction terms, and selecting "
        "the most informative signals.",
    ),
    (
        "How do you handle class imbalance?",
        "Strategies include oversampling the minority class (SMOTE), undersampling the "
        "majority class, adjusting class weights in the loss function, and using "
        "metrics like AUC-ROC instead of accuracy.",
    ),
    (
        "What is the bias-variance tradeoff?",
        "Bias is error from wrong model assumptions (underfitting); variance is error from "
        "sensitivity to training noise (overfitting). Increasing model complexity reduces "
        "bias but increases variance.",
    ),
    (
        "What is batch normalization?",
        "Normalizes layer activations to zero mean and unit variance per mini-batch, "
        "stabilizing training, allowing higher learning rates, and acting as regularization.",
    ),
    (
        "What is cross-entropy loss?",
        "Cross-entropy measures the divergence between predicted probability distributions "
        "and true labels. For classification, it penalizes confident wrong predictions "
        "more heavily than uncertain ones.",
    ),
    (
        "What is the difference between precision and recall?",
        "Precision is TP / (TP + FP): fraction of positive predictions that are correct. "
        "Recall is TP / (TP + FN): fraction of actual positives that are found.",
    ),
    (
        "What is dropout?",
        "Dropout randomly zeros a fraction of activations during training, preventing "
        "co-adaptation of neurons and acting as an ensemble of sub-networks, "
        "which reduces overfitting.",
    ),
    (
        "What is an attention mechanism?",
        "Attention computes a weighted sum of value vectors, where weights are "
        "derived from the similarity (dot product) of query and key vectors, "
        "letting the model focus on relevant positions dynamically.",
    ),
    (
        "What is transfer learning?",
        "Reusing a model pre-trained on a large dataset (e.g., ImageNet, Common Crawl) "
        "as a starting point for a related task, reducing required training data "
        "and compute.",
    ),
    (
        "What is the difference between L1 and L2 regularization?",
        "L1 (Lasso) adds the sum of absolute weights to the loss, encouraging sparsity "
        "(some weights → 0). L2 (Ridge) adds the sum of squared weights, shrinking "
        "all weights toward zero without enforcing sparsity.",
    ),
    (
        "What is gradient clipping?",
        "Scaling down gradients when their norm exceeds a threshold, preventing "
        "exploding gradients during training of deep or recurrent networks.",
    ),
    (
        "What is perplexity?",
        "Perplexity is exp(average cross-entropy loss) over a test set. It measures "
        "how well a language model predicts the next token; lower perplexity means "
        "better prediction.",
    ),
    (
        "What is the transformer architecture?",
        "A sequence model built from stacked self-attention and feed-forward layers, "
        "without recurrence. Attention allows direct token-to-token connections "
        "regardless of distance, enabling parallelism and capturing long-range dependencies.",
    ),
    (
        "What is RLHF?",
        "Reinforcement Learning from Human Feedback: a reward model trained on human "
        "preference comparisons is used to fine-tune an LLM with PPO, aligning "
        "model outputs with human values and preferences.",
    ),
    (
        "What is the difference between supervised and unsupervised learning?",
        "Supervised learning trains on labeled (input, output) pairs to predict outputs. "
        "Unsupervised learning finds structure in unlabeled data, e.g., clustering "
        "or dimensionality reduction.",
    ),
    (
        "What is knowledge distillation?",
        "Training a smaller student model to mimic the soft probability outputs of "
        "a larger teacher model, transferring knowledge while reducing inference cost.",
    ),
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(base_model: str, adapter_path: str | None, cfg: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, val_file: str, max_seq_length: int = 512) -> float:
    """Compute average perplexity over validation set examples."""
    total_loss = 0.0
    total_tokens = 0

    with open(val_file) as f:
        examples = [json.loads(line) for line in f]

    for ex in examples:
        text = (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Response:\n{ex['output']}"
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            n_tokens = inputs["input_ids"].shape[-1]

        total_loss += loss * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def qualitative_compare(base_model_path: str, adapter_path: str, cfg: dict):
    """Print side-by-side base vs fine-tuned answers on fixed eval prompts."""
    inf_cfg = cfg["inference"]

    def answer(model, tokenizer, question: str) -> str:
        # Uses module-level SYSTEM_PROMPT so this matches training exactly
        prompt = (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"### Instruction:\n{question}\n\n### Response:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=inf_cfg["max_new_tokens"],
                temperature=inf_cfg["temperature"],
                top_p=inf_cfg["top_p"],
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        new = out[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new, skip_special_tokens=True).strip()

    print("Loading base model...")
    base_model, base_tok = load_model_and_tokenizer(base_model_path, None, cfg)

    print("Loading fine-tuned model...")
    ft_model, ft_tok = load_model_and_tokenizer(base_model_path, adapter_path, cfg)

    for q in EVAL_PROMPTS:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print("\n--- BASE ---")
        print(answer(base_model, base_tok, q))
        print("\n--- FINE-TUNED ---")
        print(answer(ft_model, ft_tok, q))


def _generate_answer(model, tokenizer, question: str, cfg: dict) -> str:
    """Generate a single answer using the inference settings from config."""
    inf_cfg = cfg["inference"]
    prompt = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"### Instruction:\n{question}\n\n### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=inf_cfg["max_new_tokens"],
            temperature=inf_cfg["temperature"],
            top_p=inf_cfg["top_p"],
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def eval_task_metrics(model, tokenizer, cfg: dict) -> dict:
    """
    Compute ROUGE-L and exact-match on the curated BENCHMARK_QA pairs.

    WHY ROUGE-L over ROUGE-1:
        ROUGE-1 counts unigram overlap — a bag-of-words measure that rewards
        lists of correct keywords regardless of order. ROUGE-L measures the
        longest common subsequence, which is more sensitive to whether the
        answer is fluent and complete rather than just keyword-dense.

    WHY exact-match alongside ROUGE-L:
        ROUGE-L can score 0.6+ on answers that are close but not identical.
        Exact-match catches cases where the model outputs the reference verbatim
        (memorization vs generation) and also serves as an upper-bound sanity check.

    Returns a dict with per-question scores and summary statistics.
    """
    try:
        from rouge_score import rouge_scorer as rouge_scorer_module
    except ImportError:
        raise ImportError(
            "rouge_score not installed. Run: pip install rouge-score"
        )

    scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    results = []

    for question, reference in BENCHMARK_QA:
        prediction = _generate_answer(model, tokenizer, question, cfg)
        rouge_l = scorer.score(reference, prediction)["rougeL"].fmeasure
        exact = float(prediction.strip() == reference.strip())
        results.append({
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "rouge_l": round(rouge_l, 4),
            "exact_match": exact,
        })

    avg_rouge_l = sum(r["rouge_l"] for r in results) / len(results)
    avg_exact = sum(r["exact_match"] for r in results) / len(results)

    return {
        "avg_rouge_l": round(avg_rouge_l, 4),
        "avg_exact_match": round(avg_exact, 4),
        "n_questions": len(results),
        "per_question": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument(
        "--mode",
        choices=["perplexity", "qualitative", "metrics", "both", "all"],
        default="both",
        help=(
            "perplexity: val-set perplexity; qualitative: side-by-side answers; "
            "metrics: ROUGE-L + exact-match on benchmark; both: perplexity+qualitative; "
            "all: all three"
        ),
    )
    parser.add_argument("--config", default="config.yaml")
    # Optional: save perplexity results to JSON so rank-sweep runs can be compared later.
    # Example: python evaluate.py --adapter ./checkpoints/r8 --out results/r8.json
    parser.add_argument("--out", default=None, help="JSON path to persist results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_model = cfg["model"]["base"]

    # Pre-flight checks — fail fast before any expensive model downloads.
    validate_chat_template(cfg)
    check_gpu_memory(min_gb_required=10.0)
    validate_adapter_base_model(args.adapter, base_model)

    result: dict = {
        "adapter_path": args.adapter,
        "base_model": base_model,
        "lora_rank": cfg["lora"]["r"],
        "max_train_examples": cfg["training"].get("max_train_examples"),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if args.mode in ("perplexity", "both", "all"):
        print("Computing perplexity on validation set...")
        model, tokenizer = load_model_and_tokenizer(base_model, args.adapter, cfg)
        validate_tokenizer_template(tokenizer, cfg)
        ppl = compute_perplexity(model, tokenizer, cfg["data"]["val_file"])
        print(f"\nPerplexity (fine-tuned): {ppl:.2f}")

        # Compute base model perplexity for comparison — the improvement delta
        # is the meaningful number, not the absolute perplexity value.
        base_model_obj, base_tok = load_model_and_tokenizer(base_model, None, cfg)
        base_ppl = compute_perplexity(base_model_obj, base_tok, cfg["data"]["val_file"])
        print(f"Perplexity (base model): {base_ppl:.2f}")
        print(f"Delta: {base_ppl - ppl:+.2f} (positive = fine-tuned is better)")

        result.update({
            "perplexity_finetuned": round(ppl, 4),
            "perplexity_base": round(base_ppl, 4),
            "delta": round(base_ppl - ppl, 4),
        })

    if args.mode in ("metrics", "all"):
        print("\nComputing ROUGE-L + exact-match on benchmark Q&A...")
        model, tokenizer = load_model_and_tokenizer(base_model, args.adapter, cfg)
        validate_tokenizer_template(tokenizer, cfg)
        metrics = eval_task_metrics(model, tokenizer, cfg)
        print(f"Avg ROUGE-L:     {metrics['avg_rouge_l']:.4f}")
        print(f"Avg Exact-Match: {metrics['avg_exact_match']:.4f}")
        result["task_metrics"] = metrics

    if args.out:
        # Persist results so rank-sweep runs can be compared without re-running.
        # WHY JSON instead of just printing: terminal output is ephemeral —
        # if Colab session dies or you close the tab, results are lost.
        # Persisting to a file lets you compare e.g. r8 vs r16 vs r32 later.
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.out}")

    if args.mode in ("qualitative", "both", "all"):
        qualitative_compare(base_model, args.adapter, cfg)


if __name__ == "__main__":
    main()
