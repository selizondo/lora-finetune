# Failure Scenarios

Failure modes identified for this project. "Designed in" means a detection mechanism exists in code. "Documented gap" means the failure is understood but detection is not yet implemented.

---

## Failure 1: Adapter ↔ Base Model Version Mismatch

**What breaks:** Loading a LoRA adapter on a different base model version than it was trained on (e.g., training on `Mistral-7B-v0.1`, loading on `Mistral-7B-v0.2`). The model loads without error but produces incoherent output with no exception raised.

**Status:** Documented gap. Detection not yet implemented.

**Detection (planned):** Embed `base_model_id` in `adapter_config.json` at training time; validate at inference startup:
```python
manifest = json.load(open(f"{adapter_path}/adapter_config.json"))
if manifest["base_model_id"] != cfg["model"]["base_model"]:
    raise ValueError(f"Adapter built for {manifest['base_model_id']}, got {cfg['model']['base_model']}")
```

---

## Failure 2: Template Mismatch (Silent Quality Regression)

**What breaks:** Using a different chat template at inference than was used at training time. Mistral-7B-v0.1 expects the Llama-2 instruction format; ChatML or raw text degrades instruction-following without raising an error.

**Status:** Documented gap. Template is hardcoded in four files, no runtime validation.

**Detection (planned):** Move template to `config.yaml`; validate at startup that `tokenizer.chat_template` matches the configured template name.

---

## Failure 3: GPU OOM During Training

**What breaks:** `torch.cuda.OutOfMemoryError` mid-training if batch size or sequence length exceeds available VRAM. Process is killed; checkpoint may be corrupted or absent.

**Status:** Documented gap. No pre-flight check.

**Detection (planned):**
```python
available = torch.cuda.get_device_properties(0).total_memory
required = estimate_qlora_memory(n_params=7e9, batch_size=cfg["training"]["batch_size"],
                                  seq_len=cfg["model"]["max_seq_length"])
if required > available * 0.9:
    raise RuntimeError(f"Insufficient VRAM: need {required/1e9:.1f}GB, have {available/1e9:.1f}GB")
```
