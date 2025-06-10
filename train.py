"""
train.py — QLoRA fine-tuning with SFTTrainer.

Fine-tunes a 7-8B model using 4-bit quantization + LoRA adapters on instruction
data. Runs on a single 16GB GPU (RTX 3080/4080) or free Colab A100.

Usage:
    # 1. Prepare data first (one-time):
    python prepare_dataset.py

    # 2. Train with defaults (r=16, all examples, 3 epochs):
    python train.py

    # 3. Experiment mode — vary rank and dataset size:
    python train.py --rank 8  --examples 100  --output ./checkpoints/r8_n100
    python train.py --rank 32 --examples 500  --output ./checkpoints/r32_n500
    python train.py --rank 64 --examples 1000 --output ./checkpoints/r64_n1000

Colab A100 (free tier): ~40 min for 1000 examples, 3 epochs, r=16
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["model"]["use_nested_quant"],
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    lora = cfg["lora"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
    )


def formatting_func(example: dict) -> str:
    """Format one Alpaca-style example into the Llama-2 instruction prompt."""
    system = (
        "You are an expert ML engineer. Answer the following question clearly "
        "and concisely, as you would in a technical interview."
    )
    return (
        f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']} [/INST]</s>"
    )


def load_jsonl(path: str, max_examples: int | None = None):
    ds = load_dataset("json", data_files=path, split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None, help="Override output_dir")
    parser.add_argument("--rank", type=int, default=None, help="Override LoRA r (experiment)")
    parser.add_argument("--examples", type=int, default=None, help="Override max training examples")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides for experiment sweeps
    if args.rank is not None:
        cfg["lora"]["r"] = args.rank
        cfg["lora"]["alpha"] = args.rank * 2
    if args.examples is not None:
        cfg["training"]["max_train_examples"] = args.examples
    if args.output is not None:
        cfg["training"]["output_dir"] = args.output

    base_model = cfg["model"]["base"]
    output_dir = cfg["training"]["output_dir"]
    max_examples = cfg["training"].get("max_train_examples")

    print(f"Base model : {base_model}")
    print(f"LoRA       : r={cfg['lora']['r']}, alpha={cfg['lora']['alpha']}, "
          f"target={cfg['lora']['target_modules']}")
    print(f"Output     : {output_dir}")
    print(f"Examples   : {max_examples or 'all'}")
    print()

    # ── Model + quantization ─────────────────────────────────────────────────
    bnb_config = build_bnb_config(cfg)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for SFTTrainer with causal LM

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_cfg = build_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = load_jsonl(cfg["data"]["train_file"], max_examples)
    print(f"\nTraining on {len(train_ds)} examples")

    # ── Training args ─────────────────────────────────────────────────────────
    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"],
        warmup_ratio=t["warmup_ratio"],
        fp16=True,
        logging_steps=t["logging_steps"],
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_32bit",
        group_by_length=True,  # pads to longest seq per batch — reduces wasted FLOPS
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        formatting_func=formatting_func,
        max_seq_length=cfg["model"]["max_seq_length"],
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    final_path = os.path.join(output_dir, "final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nAdapter saved to: {final_path}")
    print(f"\nNext steps:")
    print(f"  python evaluate.py --adapter {final_path} --mode both")
    print(f"  python inference.py --adapter {final_path} --interactive")


if __name__ == "__main__":
    main()
