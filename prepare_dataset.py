"""
prepare_dataset.py — Download win-wang/Machine_Learning_QA_Collection from HuggingFace
and convert to instruction-tuning JSONL format.

The source dataset uses Gemma-style chat turns:
  <start_of_turn>user\n[question]<end_of_turn>\n<start_of_turn>model\n[answer]<end_of_turn>

This script extracts (question, answer) pairs and writes Alpaca-style JSONL.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --out data/ --max-train 5000
"""

import argparse
import json
import os
import re


DATASET_ID = "win-wang/Machine_Learning_QA_Collection"


def parse_turn(text: str) -> tuple[str, str] | None:
    """
    Extract (user_message, model_message) from a Gemma-format conversation.
    Returns None if either turn is missing or too short.
    """
    user_match = re.search(
        r"<start_of_turn>user\s*(.*?)<end_of_turn>",
        text,
        re.DOTALL,
    )
    model_match = re.search(
        r"<start_of_turn>model\s*(.*?)(?:<end_of_turn>|$)",
        text,
        re.DOTALL,
    )

    if not user_match or not model_match:
        return None

    question = user_match.group(1).strip()
    answer = model_match.group(1).strip()

    # Filter noise: skip very short or empty turns
    if len(question) < 20 or len(answer) < 30:
        return None

    return question, answer


def to_alpaca(question: str, answer: str) -> dict:
    return {
        "instruction": question,
        "input": "",
        "output": answer,
    }


def format_prompt(example: dict) -> str:
    """Format as Llama-2 instruction prompt for SFTTrainer."""
    system = (
        "You are an expert ML engineer. Answer the following question clearly "
        "and concisely, as you would in a technical interview."
    )
    return (
        f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']} [/INST]</s>"
    )


def write_jsonl(examples: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="Output directory")
    parser.add_argument(
        "--max-train", type=int, default=None,
        help="Cap training examples (default: all ~8600)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install datasets: pip install datasets")
        return

    print(f"Downloading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID)

    def process_split(split_ds, label: str, max_examples: int | None = None) -> list[dict]:
        examples = []
        skipped = 0
        for row in split_ds:
            parsed = parse_turn(row["text"])
            if parsed is None:
                skipped += 1
                continue
            question, answer = parsed
            examples.append(to_alpaca(question, answer))
            if max_examples and len(examples) >= max_examples:
                break
        print(f"  {label}: {len(examples)} examples ({skipped} skipped)")
        return examples

    train = process_split(ds["train"], "train", args.max_train)
    val = process_split(ds["validation"], "val")

    write_jsonl(train, os.path.join(args.out, "train.jsonl"))
    write_jsonl(val, os.path.join(args.out, "val.jsonl"))

    print(f"\nSaved to {args.out}/")
    print(f"  train.jsonl: {len(train)} examples")
    print(f"  val.jsonl:   {len(val)} examples")
    print("\nExample formatted prompt:")
    print("-" * 60)
    print(format_prompt(train[0])[:600])
    print("-" * 60)


if __name__ == "__main__":
    main()
