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
import math

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


EVAL_PROMPTS = [
    "What is LoRA and why is it more efficient than full fine-tuning?",
    "What is the difference between RAG and fine-tuning?",
    "What is feature engineering and why does it matter?",
    "How do you handle class imbalance in a dataset?",
    "What is the vanishing gradient problem?",
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
    SYSTEM = (
        "You are an expert ML engineer. Answer the following question clearly and concisely, "
        "as you would in a technical interview."
    )
    inf_cfg = cfg["inference"]

    def answer(model, tokenizer, question: str) -> str:
        prompt = (
            f"<s>[INST] <<SYS>>\n{SYSTEM}\n<</SYS>>\n\n"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--mode", choices=["perplexity", "qualitative", "both"], default="both")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_model = cfg["model"]["base"]

    if args.mode in ("perplexity", "both"):
        print("Computing perplexity on validation set...")
        model, tokenizer = load_model_and_tokenizer(base_model, args.adapter, cfg)
        ppl = compute_perplexity(model, tokenizer, cfg["data"]["val_file"])
        print(f"\nPerplexity (fine-tuned): {ppl:.2f}")

        # Also compute base model perplexity for comparison
        base_model_obj, base_tok = load_model_and_tokenizer(base_model, None, cfg)
        base_ppl = compute_perplexity(base_model_obj, base_tok, cfg["data"]["val_file"])
        print(f"Perplexity (base model): {base_ppl:.2f}")
        print(f"Delta: {base_ppl - ppl:+.2f} (positive = fine-tuned is better)")

    if args.mode in ("qualitative", "both"):
        qualitative_compare(base_model, args.adapter, cfg)


if __name__ == "__main__":
    main()
