"""
inference.py — Load fine-tuned LoRA adapter and generate answers.

Works locally (if you have the adapter weights) or in Colab after training.

Usage:
    python inference.py --adapter ./checkpoints/final --prompt "What is LoRA?"
    python inference.py --adapter ./checkpoints/final --interactive
    python inference.py --base mistralai/Mistral-7B-v0.1 --prompt "What is LoRA?"  # base only
"""

import argparse
import logging
import sys

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


# System prompt: must match train.py and evaluate.py exactly.
# If this diverges from training, the model generates answers in the wrong style
# (the trained format vs the prompted format mismatch causes quality regression).
SYSTEM_PROMPT = (
    "You are an expert ML engineer. Answer the following question clearly and concisely, "
    "as you would in a technical interview."
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(base_model: str, adapter_path: str | None, cfg: dict):
    """Load quantized base model + optional LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["model"]["use_nested_quant"],
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
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        # merge_and_unload() folds the LoRA delta matrices into the base weights
        # and removes the PEFT wrapper. WHY: during inference we never need to
        # update the LoRA matrices — merging eliminates the extra forward pass
        # through the LoRA branch, giving ~15% faster token generation.
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def build_prompt(question: str) -> str:
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"### Instruction:\n{question}\n\n### Response:\n"
    )


def generate(model, tokenizer, prompt: str, cfg: dict) -> str:
    inf_cfg = cfg["inference"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=inf_cfg["max_new_tokens"],
            temperature=inf_cfg["temperature"],
            top_p=inf_cfg["top_p"],
            repetition_penalty=inf_cfg["repetition_penalty"],
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the input tokens from the output
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--base", default=None, help="Base model override (default: from config.yaml)")
    parser.add_argument("--prompt", default=None, help="Single question to answer")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A loop")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_model = args.base or cfg["model"]["base"]

    # Pre-flight checks (Items 1, 2, 3) — fail fast before any expensive downloads.
    validate_chat_template(cfg)
    check_gpu_memory(min_gb_required=10.0)
    if args.adapter:
        validate_adapter_base_model(args.adapter, base_model)

    print(f"Loading model: {base_model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    model, tokenizer = load_model(base_model, args.adapter, cfg)
    validate_tokenizer_template(tokenizer, cfg)

    if args.prompt:
        prompt = build_prompt(args.prompt)
        answer = generate(model, tokenizer, prompt, cfg)
        print(f"\nQ: {args.prompt}\nA: {answer}")

    elif args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            question = input("Q: ").strip()
            if question.lower() in {"quit", "exit", "q"}:
                break
            if not question:
                continue
            prompt = build_prompt(question)
            answer = generate(model, tokenizer, prompt, cfg)
            print(f"A: {answer}\n")

    else:
        print("Provide --prompt or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
