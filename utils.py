"""
utils.py — Shared validation helpers for lora-finetune.

Functions here are imported by train.py, evaluate.py, and inference.py.
Keeping them in one module means a fix propagates to all entry-points at once
and makes them trivially testable without loading a model.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

# torch is optional: utils.py may be imported in test environments that don't
# have CUDA drivers.  All GPU-dependent code checks `torch` at call time.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Item 1: Adapter ↔ base model version guard
# ---------------------------------------------------------------------------

def validate_adapter_base_model(adapter_path: str, expected_base_model_id: str) -> None:
    """
    Verify that the adapter stored at adapter_path was trained on expected_base_model_id.

    PEFT writes the original base model path into adapter_config.json under the key
    'base_model_name_or_path'.  If this doesn't match the model we're about to load,
    we'll get nonsense output rather than an error — a silent failure mode that is
    very hard to debug after the fact.

    Raises ValueError if the IDs don't match.
    Raises FileNotFoundError if adapter_config.json is absent (corrupt adapter dir).
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found at {config_path}. "
            "The adapter directory may be incomplete or the path is wrong."
        )

    with open(config_path) as f:
        adapter_cfg = json.load(f)

    # PEFT uses 'base_model_name_or_path' as the canonical field name.
    stored_base = adapter_cfg.get("base_model_name_or_path")
    if stored_base is None:
        logging.warning(
            "adapter_config.json has no 'base_model_name_or_path' field. "
            "Cannot verify adapter ↔ base model compatibility. "
            "Confirm manually that the adapter was trained on: %s",
            expected_base_model_id,
        )
        return

    if stored_base != expected_base_model_id:
        raise ValueError(
            f"Adapter/base model mismatch: adapter was trained on '{stored_base}' "
            f"but current config specifies '{expected_base_model_id}'. "
            "Either point --adapter at the correct adapter or update model.base in config.yaml."
        )

    logging.info("Adapter base model verified: %s", stored_base)


# ---------------------------------------------------------------------------
# Item 2: Chat template validation
# ---------------------------------------------------------------------------

# Template signatures: substring that must appear in the tokenizer's chat_template
# if one is set.  For models without a built-in chat_template (Mistral-7B-v0.1),
# we validate the config value is one of the known-good strings instead.
_KNOWN_TEMPLATES = {
    "llama_2": "[INST]",
    "chatml": "<|im_start|>",
    "alpaca": "### Instruction",
}

# Required format placeholders in the Alpaca/Llama-2 template strings we build.
# These must appear in the formatted string so that all fields are substituted.
_TEMPLATE_REQUIRED_KEYS = {"instruction", "output"}


def validate_chat_template(cfg: dict) -> None:
    """
    Confirm that config.yaml specifies a recognised chat_template value.

    WHY: if the template name in config.yaml is misspelled or swapped to a
    different model family, every formatted example will be silently wrong.
    This check costs nothing at startup and catches the failure before
    the training/eval run begins.

    Does NOT require a live tokenizer — checks the config value only.
    If the tokenizer has a built-in chat_template, callers may additionally
    compare it against the config value (see validate_tokenizer_template).
    """
    template_name = cfg.get("model", {}).get("chat_template")
    if template_name is None:
        raise ValueError(
            "config.yaml is missing model.chat_template. "
            "Add 'chat_template: llama_2' (or the appropriate value) under the model key."
        )

    if template_name not in _KNOWN_TEMPLATES:
        raise ValueError(
            f"Unknown chat_template '{template_name}' in config.yaml. "
            f"Known values: {list(_KNOWN_TEMPLATES.keys())}"
        )

    logging.info("Chat template config validated: %s", template_name)


def validate_tokenizer_template(tokenizer, cfg: dict) -> None:
    """
    If the tokenizer ships with a built-in chat_template, verify it matches
    the template family declared in config.yaml.

    Mistral-7B-v0.1 does not set tokenizer.chat_template, so this is a no-op
    for the default config.  It becomes relevant if the base model is swapped.
    """
    if not getattr(tokenizer, "chat_template", None):
        return  # tokenizer has no built-in template to check against

    template_name = cfg["model"]["chat_template"]
    signature = _KNOWN_TEMPLATES[template_name]
    if signature not in tokenizer.chat_template:
        raise ValueError(
            f"Tokenizer's built-in chat_template does not match config value '{template_name}'. "
            f"Expected to find '{signature}' in tokenizer.chat_template. "
            "Update model.chat_template in config.yaml to match the actual tokenizer."
        )

    logging.info("Tokenizer chat_template matches config: %s", template_name)


# ---------------------------------------------------------------------------
# Item 3: GPU OOM pre-flight check
# ---------------------------------------------------------------------------

def check_gpu_memory(min_gb_required: float = 10.0) -> None:
    """
    Warn (non-fatal) if the GPU doesn't have enough free memory for QLoRA.

    WHY non-fatal: Colab T4 has 15GB total; after OS/driver overhead ~14GB are
    available.  At QLoRA with batch_size=4 and seq_len=512 the peak is ~13–14GB,
    so a pre-flight warning at <10GB free is a signal something is already loaded
    in VRAM rather than an unconditional blocker.  Raising here would break the
    normal Colab workflow where the user might re-run just this cell.

    torch is imported at module level (with a try/except for environments without
    CUDA drivers).  Tests mock 'utils.torch' to exercise all code paths without
    a real GPU.
    """
    if torch is None:
        logging.warning("torch not available — skipping GPU memory pre-flight check.")
        return

    if not torch.cuda.is_available():
        logging.warning(
            "No CUDA GPU detected. Training on CPU will be extremely slow. "
            "Run on Colab T4/A100 or a machine with an NVIDIA GPU."
        )
        return

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    allocated_gb = torch.cuda.memory_allocated(0) / 1e9
    free_gb = total_gb - allocated_gb

    logging.info(
        "GPU: %s | Total: %.1f GB | Allocated: %.1f GB | Free: %.1f GB",
        props.name, total_gb, allocated_gb, free_gb,
    )

    if free_gb < min_gb_required:
        logging.warning(
            "GPU has %.1f GB free; QLoRA fine-tuning of a 7B model requires ~%.1f GB. "
            "OOM is likely. Free VRAM before proceeding (restart runtime or reduce batch_size).",
            free_gb, min_gb_required,
        )
