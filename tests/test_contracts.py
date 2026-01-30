"""
tests/test_contracts.py — Contract tests for lora-finetune.

These tests verify system-level contracts that are documented in prose
(STAFF_REVIEW.md, tradeoffs.md) but were previously unchecked by code.
Contracts tested:
  1. Adapter/base mismatch raises ValueError at inference time.
  2. GPU memory check emits a WARNING when free VRAM is below the threshold.
  3. config.yaml contains all required keys and recognised values.

WHY contract tests rather than unit tests:
    Unit tests verify internal logic; contract tests verify the boundary
    behaviour that the rest of the system depends on. If validate_adapter_base_model()
    silently passes on a mismatch, every downstream inference is wrong — that's a
    contract failure, not a logic error.  These tests catch that at CI time.

Run with:
    pytest tests/test_contracts.py -v
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Ensure repo root is on sys.path so we can import utils without installing.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import (  # noqa: E402
    check_gpu_memory,
    validate_adapter_base_model,
    validate_chat_template,
    validate_tokenizer_template,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter_dir(base_model_name: str) -> str:
    """
    Write a minimal adapter_config.json into a temp directory and return its path.
    PEFT uses 'base_model_name_or_path' as the key.
    """
    d = tempfile.mkdtemp()
    cfg = {
        "base_model_name_or_path": base_model_name,
        "r": 16,
        "peft_type": "LORA",
    }
    with open(os.path.join(d, "adapter_config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def _make_config(overrides: dict | None = None) -> dict:
    """Return a minimal valid config dict, with optional key overrides."""
    base = {
        "model": {
            "base": "mistralai/Mistral-7B-v0.1",
            "chat_template": "llama_2",
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "use_nested_quant": False,
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                 "target_modules": ["q_proj", "v_proj"], "bias": "none",
                 "task_type": "CAUSAL_LM"},
        "training": {
            "output_dir": "./checkpoints",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "weight_decay": 0.001,
            "fp16": True,
            "bf16": False,
            "max_grad_norm": 0.3,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "load_best_model_at_end": True,
            "report_to": "none",
        },
        "data": {
            "train_file": "data/train.jsonl",
            "val_file": "data/val.jsonl",
            "max_seq_length": 512,
            "val_split": 0.1,
        },
        "inference": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
    }
    if overrides:
        for key, val in overrides.items():
            base[key].update(val)
    return base


# ---------------------------------------------------------------------------
# Contract 1: Adapter ↔ base model version guard
# ---------------------------------------------------------------------------

class TestAdapterVersionGuard:
    """validate_adapter_base_model must raise ValueError on ID mismatch."""

    def test_matching_base_model_passes(self):
        """No error when adapter and config specify the same base model."""
        adapter_dir = _make_adapter_dir("mistralai/Mistral-7B-v0.1")
        # Should not raise
        validate_adapter_base_model(adapter_dir, "mistralai/Mistral-7B-v0.1")

    def test_mismatched_base_model_raises(self):
        """
        Core contract: a wrong base model ID must raise ValueError.
        If this silently passes, every inference on the mismatched adapter is wrong.
        """
        adapter_dir = _make_adapter_dir("mistralai/Mistral-7B-v0.1")
        with pytest.raises(ValueError, match="Adapter/base model mismatch"):
            validate_adapter_base_model(adapter_dir, "mistralai/Mistral-7B-v0.2")

    def test_mismatch_error_names_both_models(self):
        """Error message must name both the stored and expected model for debuggability."""
        adapter_dir = _make_adapter_dir("mistralai/Mistral-7B-v0.1")
        with pytest.raises(ValueError) as exc_info:
            validate_adapter_base_model(adapter_dir, "wrong-model-id")
        msg = str(exc_info.value)
        assert "mistralai/Mistral-7B-v0.1" in msg
        assert "wrong-model-id" in msg

    def test_missing_adapter_config_raises_file_not_found(self):
        """Corrupt adapter dir (no adapter_config.json) must fail explicitly."""
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(FileNotFoundError):
                validate_adapter_base_model(d, "any-model")

    def test_adapter_config_without_base_model_field_warns(self, caplog):
        """
        If adapter_config.json has no 'base_model_name_or_path' key, emit a
        WARNING rather than silently passing — the caller can't assume compatibility.
        """
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump({"r": 16, "peft_type": "LORA"}, f)  # no base_model_name_or_path

        with caplog.at_level(logging.WARNING, logger="root"):
            validate_adapter_base_model(d, "some-model")

        assert any("base_model_name_or_path" in r.message for r in caplog.records), (
            "Expected a warning about missing 'base_model_name_or_path' field"
        )


# ---------------------------------------------------------------------------
# Contract 2: GPU memory pre-flight check
# ---------------------------------------------------------------------------

class TestGpuMemoryCheck:
    """check_gpu_memory must warn when free VRAM is below the threshold."""

    def _make_gpu_props(self, total_bytes: int) -> SimpleNamespace:
        """Return a fake torch.cuda.DeviceProperties-like object."""
        props = SimpleNamespace()
        props.total_memory = total_bytes
        props.name = "NVIDIA Fake GPU"
        return props

    def test_sufficient_memory_no_warning(self, caplog):
        """14 GB free — no warning should be emitted."""
        total = int(14 * 1e9)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = self._make_gpu_props(total)
        mock_torch.cuda.memory_allocated.return_value = 0

        with (
            patch("utils.torch", mock_torch),
            caplog.at_level(logging.WARNING, logger="root"),
        ):
            check_gpu_memory(min_gb_required=10.0)

        assert not any(
            "OOM" in r.message or "free" in r.message.lower()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        ), "Unexpected OOM warning with sufficient memory"

    def test_low_memory_emits_warning(self, caplog):
        """
        Core contract: when free VRAM is below threshold, a WARNING must be logged.
        This is non-fatal (the caller decides whether to abort) but must be visible.
        """
        total = int(8 * 1e9)    # 8 GB total
        allocated = int(1 * 1e9)  # 1 GB already used → 7 GB free < 10 GB threshold
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = self._make_gpu_props(total)
        mock_torch.cuda.memory_allocated.return_value = allocated

        with (
            patch("utils.torch", mock_torch),
            caplog.at_level(logging.WARNING, logger="root"),
        ):
            check_gpu_memory(min_gb_required=10.0)

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OOM" in m or "free" in m.lower() for m in warning_messages), (
            f"Expected OOM/free-memory warning. Got: {warning_messages}"
        )

    def test_no_gpu_emits_warning(self, caplog):
        """No CUDA GPU detected must emit a WARNING (not silently succeed)."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with (
            patch("utils.torch", mock_torch),
            caplog.at_level(logging.WARNING, logger="root"),
        ):
            check_gpu_memory()

        assert any(
            "No CUDA" in r.message or "GPU" in r.message
            for r in caplog.records
            if r.levelno >= logging.WARNING
        ), "Expected a warning when no GPU is detected"

    def test_gpu_check_is_non_fatal(self):
        """check_gpu_memory must not raise even when VRAM is critically low."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = self._make_gpu_props(int(1e9))
        mock_torch.cuda.memory_allocated.return_value = 0

        with patch("utils.torch", mock_torch):
            # Must not raise
            check_gpu_memory(min_gb_required=10.0)


# ---------------------------------------------------------------------------
# Contract 3: Config schema validation
# ---------------------------------------------------------------------------

class TestConfigSchema:
    """config.yaml must have all required keys with valid values."""

    _CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")

    # Required top-level sections and their required sub-keys.
    _REQUIRED_STRUCTURE = {
        "model": ["base", "chat_template", "load_in_4bit",
                  "bnb_4bit_quant_type", "use_nested_quant"],
        "lora": ["r", "alpha", "dropout", "target_modules"],
        "training": ["output_dir", "num_train_epochs",
                     "per_device_train_batch_size", "learning_rate"],
        "data": ["train_file", "val_file", "max_seq_length"],
        "inference": ["max_new_tokens", "temperature", "top_p"],
    }

    def _load_config(self) -> dict:
        assert os.path.exists(self._CONFIG_PATH), (
            f"config.yaml not found at {self._CONFIG_PATH}"
        )
        with open(self._CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def test_config_file_exists(self):
        assert os.path.exists(self._CONFIG_PATH)

    def test_required_top_level_sections_present(self):
        cfg = self._load_config()
        for section in self._REQUIRED_STRUCTURE:
            assert section in cfg, f"Missing top-level section: '{section}'"

    def test_required_sub_keys_present(self):
        cfg = self._load_config()
        for section, keys in self._REQUIRED_STRUCTURE.items():
            for key in keys:
                assert key in cfg[section], (
                    f"Missing key '{key}' in config.yaml section '{section}'"
                )

    def test_chat_template_is_valid(self):
        """validate_chat_template must pass on the actual config.yaml."""
        cfg = self._load_config()
        # Should not raise
        validate_chat_template(cfg)

    def test_chat_template_key_present(self):
        cfg = self._load_config()
        assert "chat_template" in cfg["model"], (
            "model.chat_template is required in config.yaml (Item 2)"
        )

    def test_base_model_id_is_string(self):
        cfg = self._load_config()
        assert isinstance(cfg["model"]["base"], str) and cfg["model"]["base"], (
            "model.base must be a non-empty string"
        )

    def test_max_seq_length_positive(self):
        cfg = self._load_config()
        assert cfg["data"]["max_seq_length"] > 0

    def test_lora_rank_positive(self):
        cfg = self._load_config()
        assert cfg["lora"]["r"] > 0

    def test_validate_chat_template_rejects_unknown_value(self):
        """validate_chat_template must raise on unrecognised template names."""
        bad_cfg = _make_config({"model": {"chat_template": "unknown_template"}})
        with pytest.raises(ValueError, match="Unknown chat_template"):
            validate_chat_template(bad_cfg)

    def test_validate_chat_template_rejects_missing_key(self):
        """validate_chat_template must raise when model.chat_template is absent."""
        cfg = _make_config()
        del cfg["model"]["chat_template"]
        with pytest.raises(ValueError, match="missing model.chat_template"):
            validate_chat_template(cfg)
