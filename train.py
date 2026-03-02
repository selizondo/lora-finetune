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
import logging
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

import subprocess

from utils import check_gpu_memory, validate_chat_template, validate_tokenizer_template

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _s3_sync(project: str, local_dir: str, artifact_type: str) -> None:
    """Sync local artifacts to S3 if S3_ARTIFACTS_BUCKET is set. Non-fatal on failure."""
    bucket = os.getenv("S3_ARTIFACTS_BUCKET", "travissketch-portfolio-artifacts")
    s3_path = f"s3://{bucket}/{project}/{artifact_type}/"
    print(f"\nSyncing {artifact_type} → {s3_path}")
    result = subprocess.run(
        ["aws", "s3", "sync", local_dir + "/", s3_path,
         "--exclude", "*.pyc", "--exclude", "__pycache__/*"],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[warn] S3 sync failed (non-fatal) — artifacts remain at {local_dir}")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    """
    Build the BitsAndBytes 4-bit quantization config.

    WHY 4-bit quantization (QLoRA):
        Mistral-7B has ~7B parameters. At fp16 (2 bytes each) that's ~14GB —
        too large for a free Colab T4 (16GB VRAM). 4-bit quantization compresses
        each weight to 4 bits, reducing memory to ~3.5GB and leaving room for
        activations, gradients, and optimizer state.

    WHY nf4 (NormalFloat 4-bit) over int4:
        nf4 is optimised for normally-distributed weights (which neural networks
        have). It minimises quantization error compared to linear int4 binning,
        recovering ~1-2 perplexity points at the same memory cost.

    WHY use_nested_quant (double quantization):
        Quantizes the quantization constants themselves, saving an additional
        ~0.4 bits per parameter — about 400MB for a 7B model. Small but free.
    """
    return BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["model"]["use_nested_quant"],
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    """
    Build the LoRA adapter configuration.

    WHY LoRA instead of full fine-tuning:
        Full fine-tuning updates all ~7B parameters — requires ~56GB VRAM and
        hours of compute. LoRA inserts small trainable matrices (rank r=16)
        into the attention layers, training only ~0.5% of parameters while
        achieving >90% of the quality gain. We can train on a single T4/A100.

    WHY r=16 (LoRA rank):
        Rank controls the size of the inserted matrices. r=8 is faster but
        loses more quality on domain-specific tasks. r=32+ gives diminishing
        returns while using 2x the parameters. r=16 is the standard default
        for instruction-tuning tasks of this size (~8K examples).

    WHY alpha = r * 2:
        lora_alpha is a scaling factor applied to the LoRA output before adding
        it to the base weight. Setting alpha = 2r is a standard heuristic that
        keeps the LoRA contribution appropriately scaled regardless of rank —
        changing r without changing alpha would alter effective learning rate.

    WHY target_modules = attention projections (q_proj, v_proj, etc.):
        Attention layers capture most of the model's "knowledge" about style and
        format. Targeting only these (not MLP layers) gives the best quality/
        parameter tradeoff for instruction-following tasks.
    """
    lora = cfg["lora"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
    )


# System prompt used in every training example AND at inference time.
# WHY defined at module level (not inline in formatting_func):
#   It must match the prompt used in evaluate.py and inference.py.
#   If they diverge, the model was trained expecting one framing and gets
#   asked questions with a different one — silent quality regression.
SYSTEM_PROMPT = (
    "You are an expert ML engineer. Answer the following question clearly "
    "and concisely, as you would in a technical interview."
)


def formatting_func(example: dict) -> str:
    """
    Format one Alpaca-style example into the Llama-2 instruction prompt.

    WHY Llama-2 chat template (<s>[INST] <<SYS>>...):
        Mistral-7B-v0.1 was instruction-tuned with this exact template.
        Using plain text or ChatML means the model doesn't recognise the
        system block as instructions — it just treats the whole thing as
        continuable text. Template must match between training and inference.

    The [/INST]</s> closing tag signals end-of-answer during generation.
    """
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
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

    # Validate config before any expensive work (Items 2 & 3).
    validate_chat_template(cfg)
    check_gpu_memory(min_gb_required=10.0)

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
    validate_tokenizer_template(tokenizer, cfg)  # no-op if tokenizer has no built-in template

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
    # Pre-format so trl 1.x reads a concrete text column (formatting_func deprecated).
    train_ds = train_ds.map(lambda ex: {"text": formatting_func(ex)}, remove_columns=train_ds.column_names)
    print(f"\nTraining on {len(train_ds)} examples")

    # ── Training args ─────────────────────────────────────────────────────────
    t = cfg["training"]
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=t["num_train_epochs"],
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            learning_rate=t["learning_rate"],
            lr_scheduler_type=t["lr_scheduler_type"],
            warmup_ratio=t["warmup_ratio"],
            fp16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=t["logging_steps"],
            save_strategy=t.get("save_strategy", "epoch"),
            save_total_limit=2,
            report_to="none",
            optim="paged_adamw_32bit",
            dataset_text_field="text",
            max_seq_length=cfg["data"]["max_seq_length"],
        ),
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    final_path = os.path.join(output_dir, "final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nAdapter saved to: {final_path}")

    _s3_sync("lora-finetune", output_dir, "adapter")

    print(f"\nNext steps:")
    print(f"  python evaluate.py --adapter {final_path} --mode both")
    print(f"  python inference.py --adapter {final_path} --interactive")


if __name__ == "__main__":
    main()
