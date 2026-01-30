"""
baseline/bm25_baseline.py — BM25 retrieval baseline for ML Q&A.

WHY this exists:
    The README claims fine-tuning achieves ~65% accuracy vs ~70% for RAG on this
    benchmark.  That claim lives only in prose — this script makes it verifiable
    in code.  Running this on the same BENCHMARK_QA pairs as evaluate.py lets you
    compare ROUGE-L side-by-side between the fine-tuned model and pure retrieval.

HOW the retrieval baseline works:
    For each test question, BM25 ranks all training examples by query-document
    similarity and returns the top-1 answer.  This simulates a retrieval-only
    system with no generative step — the "answer" is literally the closest training
    example's answer.  It's a strong, honest baseline: if the fine-tuned model
    can't beat BM25 on ROUGE-L, the fine-tuning added no value for this task.

NOTE: This script does not run inference through the LLM.  It is purely a
retrieval system.  Run it alongside evaluate.py --mode metrics to compare scores.

Usage:
    python baseline/bm25_baseline.py --train data/train.jsonl
    python baseline/bm25_baseline.py --train data/train.jsonl --out results/bm25.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Benchmark Q&A pairs (same as evaluate.py — single source of truth is
# evaluate.py; this import avoids duplication).
# ---------------------------------------------------------------------------
# The script can be run standalone (from the repo root) or from baseline/.
# Add repo root to sys.path so the import always resolves correctly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluate import BENCHMARK_QA  # noqa: E402


def load_train_data(train_file: str) -> List[dict]:
    """Load all examples from the training JSONL file."""
    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training file not found: {train_file}\n"
            "Run: python prepare_dataset.py  (to generate data/train.jsonl)"
        )
    with open(train_file, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_bm25_index(examples: List[dict]):
    """
    Build a BM25 index over training questions.

    WHY BM25 over TF-IDF or embedding similarity:
        BM25 is a bag-of-words retrieval function that accounts for term
        frequency saturation and document length normalization.  It's the
        standard sparse retrieval baseline used in academic NLP benchmarks
        (MS MARCO, Natural Questions) and is what production search engines
        (Elasticsearch, OpenSearch) implement under the hood.  A fine-tuned
        model should outperform BM25 if the task requires reasoning beyond
        pattern matching.

    Returns:
        (bm25_index, questions, answers) tuple.
        questions[i] and answers[i] correspond to the i-th document in the index.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError(
            "rank_bm25 not installed. Run: pip install rank-bm25"
        )

    questions = [ex["instruction"] for ex in examples]
    answers = [ex["output"] for ex in examples]
    # Tokenize by whitespace — fast and sufficient for BM25 on English text.
    tokenized = [q.lower().split() for q in questions]
    return BM25Okapi(tokenized), questions, answers


def retrieve_top1(bm25_index, query: str, answers: List[str]) -> str:
    """Return the answer corresponding to the highest-scoring training example."""
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_idx = int(scores.argmax())
    return answers[top_idx]


def compute_rouge_l(reference: str, prediction: str) -> float:
    """Compute ROUGE-L F-measure between reference and prediction."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge_score not installed. Run: pip install rouge-score"
        )
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def run_bm25_baseline(
    train_file: str,
    benchmark: List[Tuple[str, str]],
) -> dict:
    """
    Run BM25 retrieval on all benchmark questions and compute ROUGE-L + exact-match.

    Returns a dict with per-question scores and summary statistics, in the same
    format as evaluate.eval_task_metrics() so results can be compared directly.
    """
    print(f"Loading training data from: {train_file}")
    examples = load_train_data(train_file)
    print(f"Indexed {len(examples)} training examples.")

    bm25_index, _, answers = build_bm25_index(examples)

    results = []
    for question, reference in benchmark:
        prediction = retrieve_top1(bm25_index, question, answers)
        rouge_l = compute_rouge_l(reference, prediction)
        exact = float(prediction.strip() == reference.strip())
        results.append({
            "question": question,
            "reference": reference,
            "retrieved_answer": prediction,
            "rouge_l": round(rouge_l, 4),
            "exact_match": exact,
        })

    avg_rouge_l = sum(r["rouge_l"] for r in results) / len(results)
    avg_exact = sum(r["exact_match"] for r in results) / len(results)

    return {
        "method": "bm25_top1",
        "train_file": train_file,
        "n_train_examples": len(examples),
        "n_questions": len(results),
        "avg_rouge_l": round(avg_rouge_l, 4),
        "avg_exact_match": round(avg_exact, 4),
        "per_question": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BM25 retrieval baseline — compare against fine-tuned model metrics"
    )
    parser.add_argument(
        "--train",
        default="data/train.jsonl",
        help="Path to training JSONL (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON path to persist results (e.g. results/bm25.json)",
    )
    args = parser.parse_args()

    summary = run_bm25_baseline(args.train, BENCHMARK_QA)

    print(f"\n{'=' * 60}")
    print("BM25 Retrieval Baseline Results")
    print(f"{'=' * 60}")
    print(f"Train examples indexed : {summary['n_train_examples']}")
    print(f"Benchmark questions    : {summary['n_questions']}")
    print(f"Avg ROUGE-L            : {summary['avg_rouge_l']:.4f}")
    print(f"Avg Exact-Match        : {summary['avg_exact_match']:.4f}")
    print()
    print("Compare these scores against:")
    print("  python evaluate.py --adapter ./checkpoints/final --mode metrics --out results/ft.json")
    print()
    print("Expected: fine-tuned model should match or exceed BM25 ROUGE-L.")
    print("README claim: FT ~65% accuracy vs RAG ~70% — this script lets you verify.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
