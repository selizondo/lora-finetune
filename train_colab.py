#!/usr/bin/env python
# coding: utf-8

# # LoRA Fine-Tuning — Colab Experiment
# 
# QLoRA fine-tuning of GPT-Neo 2.7B on ML interview Q&A.
# 
# **Covers:**
# 1. Data preparation (`win-wang/Machine_Learning_QA_Collection`)
# 2. Baseline training (r=16, full dataset — ~80 min on T4)
# 3. Rank sweep: r=8 / r=16 / r=64 at 1000 examples
# 4. Data scaling: n=100 / n=500 / n=1000 at r=16
# 5. Perplexity evaluation (base vs fine-tuned)
# 6. Qualitative side-by-side comparison
# 7. Save to Google Drive or download
# 
# **Required runtime:** T4 or better — free Colab default. No runtime change needed.
# 
# **HuggingFace token:** Not required for this public model.
# 

# In[ ]:


import subprocess, sys
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
    capture_output=True, text=True
)
if result.returncode == 0:
    print('GPU:', result.stdout.strip())
else:
    raise RuntimeError('No GPU detected — Runtime > Change runtime type > GPU > T4')


# In[ ]:


import subprocess, sys
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'transformers', 'datasets', 'peft', 'trl', 'bitsandbytes',
    'accelerate', 'huggingface_hub', 'pyyaml'
], check=False)
print('Packages ready.')


# In[4]:


import trl
print(f"trl version: {trl.__version__}")


# If you encounter a `TypeError` when using `SFTTrainer`, it might be due to an outdated `trl` library. You can upgrade `trl` to the latest version by running the following cell.

# In[5]:


# Uncomment the line below and run this cell to upgrade trl if needed.
# %pip install --upgrade trl


# In[6]:


import os, shutil

# Check if running in Colab
try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount('/content/drive')
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    # ── Persistent paths (survive disconnect) ────────────────────────────────────
    DRIVE_BASE    = '/content/drive/MyDrive/git_repos/lora-finetune'
    DATASET_DIR   = '/content/drive/MyDrive/datasets/lora-finetune'
    DRIVE_RESULTS = DRIVE_BASE    + '/results'

    # ── Ephemeral local paths (fast I/O during training) ─────────────────────────
    LOCAL_DATA    = '/content/data'
    LOCAL_CKPT    = '/content/checkpoints'
else:
    # EC2 / macOS / local Unix paths.
    # Override RESULTS_DIR env var to redirect output (e.g. RESULTS_DIR=/home/ubuntu/lora-finetune).
    import pathlib
    DRIVE_BASE    = os.environ.get('RESULTS_DIR', str(pathlib.Path(__file__).resolve().parent))
    DATASET_DIR   = DRIVE_BASE + '/datasets'
    DRIVE_RESULTS = DRIVE_BASE + '/results'
    LOCAL_DATA    = DRIVE_BASE + '/data'
    LOCAL_CKPT    = DRIVE_BASE + '/checkpoints'

TRAIN_FILE    = LOCAL_DATA + '/train.jsonl'
VAL_FILE      = LOCAL_DATA + '/val.jsonl'

for d in [DATASET_DIR, DRIVE_RESULTS, LOCAL_DATA, LOCAL_CKPT]:
    os.makedirs(d, exist_ok=True)

print('Setup complete.')
print('Running in:', 'Colab' if IN_COLAB else 'Local')
print('  DATASET_DIR   :', DATASET_DIR)
print('  DRIVE_RESULTS :', DRIVE_RESULTS)
print('  LOCAL_DATA    :', LOCAL_DATA)
print('  LOCAL_CKPT    :', LOCAL_CKPT)


# In[ ]:


from huggingface_hub import login
import os

try:
    from google.colab import userdata
    login(token=userdata.get('HF_TOKEN'), add_to_git_credential=False)
    print('Logged in from Colab Secrets')
except ImportError:
    token = os.environ.get('HF_TOKEN')
    if token:
        login(token=token, add_to_git_credential=False)
        print('Logged in from HF_TOKEN env var')
    else:
        print('No HF_TOKEN — skipping login (public models only)')


# ## Data Preparation
# 
# Downloads `win-wang/Machine_Learning_QA_Collection` (~8600 Gemma-format conversations),
# converts to Alpaca JSONL, writes `train.jsonl` and `val.jsonl` (90/10 split).
# 
# Quality filters: skip questions < 20 chars, answers < 30 chars.

# In[8]:


import json, re, shutil
from datasets import load_dataset

drive_train = DATASET_DIR + '/train.jsonl'
drive_val   = DATASET_DIR + '/val.jsonl'

if os.path.exists(drive_train) and os.path.exists(drive_val):
    print('Found JSONL on Drive — copying to local...')
    shutil.copy(drive_train, TRAIN_FILE)
    shutil.copy(drive_val,   VAL_FILE)
    n_train = sum(1 for _ in open(TRAIN_FILE))
    n_val   = sum(1 for _ in open(VAL_FILE))
    print(f'Train: {n_train}, Val: {n_val} (loaded from Drive, skipping download)')
else:
    def parse_gemma(text):
        u = re.search(r'<start_of_turn>user\s*(.*?)<end_of_turn>', text, re.DOTALL)
        m = re.search(r'<start_of_turn>model\s*(.*?)(?:<end_of_turn>|$)', text, re.DOTALL)
        if not u or not m:
            return None
        q, a = u.group(1).strip(), m.group(1).strip()
        return (q, a) if len(q) >= 20 and len(a) >= 30 else None

    def write_jsonl(examples, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

    print('Downloading win-wang/Machine_Learning_QA_Collection...')
    ds = load_dataset('win-wang/Machine_Learning_QA_Collection')

    examples, skipped = [], 0
    for row in ds['train']:
        p = parse_gemma(row['text'])
        if p:
            examples.append({'instruction': p[0], 'input': '', 'output': p[1]})
        else:
            skipped += 1

    n_val = max(50, len(examples) // 10)
    train_data, val_data = examples[:-n_val], examples[-n_val:]

    write_jsonl(train_data, TRAIN_FILE)
    write_jsonl(val_data,   VAL_FILE)

    # Persist to Drive — skip download on future runs
    shutil.copy(TRAIN_FILE, drive_train)
    shutil.copy(VAL_FILE,   drive_val)

    print(f'Train: {len(train_data)}, Val: {len(val_data)}, Skipped: {skipped}')
    print(f'Saved to Drive: {DATASET_DIR}')

print('\nSample instruction:', open(TRAIN_FILE).readline()[:120])


# ## Configuration & Helpers
# 
# - `load_model(rank)` — loads 4-bit GPT-Neo 2.7B with LoRA adapters (0.5% of params trained)
# - `run_train(model, tok, out_dir, n, epochs)` — SFTTrainer, saves adapter to `out_dir/final`
# - `compute_ppl(model, tok)` — average perplexity on val set (lower = better fit)
# 

# In[9]:


import os
os.environ['PYTHONUTF8'] = '1'
print('UTF-8 encoding set')


# In[ ]:


import math, torch

# Fix for Windows UTF-8 issue in trl
import pathlib
original_read_text = pathlib.Path.read_text
pathlib.Path.read_text = lambda self, encoding='utf-8', errors=None, newline=None: original_read_text(self, encoding=encoding, errors=errors, newline=newline)

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, TaskType, get_peft_model,
    prepare_model_for_kbit_training, PeftModel
)
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset as lds

BASE_MODEL = 'EleutherAI/gpt-neo-1.3B'
SYSTEM = (
    'You are an expert ML engineer. Answer the following question clearly '
    'and concisely, as you would in a technical interview.'
)

def fmt(ex):
    return (
        '<s>[INST] <<SYS>>\n' + SYSTEM + '\n<</SYS>>\n\n'
        + '### Instruction:\n' + ex['instruction'] + '\n\n'
        + '### Response:\n' + ex['output'] + ' [/INST]</s>'
    )

def load_model(rank=16):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map='auto',
        dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank * 2,
        lora_dropout=0.05, target_modules=['q_proj', 'v_proj'], bias='none',
    ))
    model.print_trainable_parameters()
    return model, tok

def run_train(model, tok, out_dir, n=None, epochs=3):
    ds = lds('json', data_files=TRAIN_FILE, split='train')
    if n:
        ds = ds.select(range(min(n, len(ds))))
    # Pre-apply formatting so trl 1.4 can read a concrete text column
    ds = ds.map(lambda ex: {'text': fmt(ex)}, remove_columns=ds.column_names)
    print(f'  Training on {len(ds)} examples for {epochs} epochs...')
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=out_dir, num_train_epochs=epochs,
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            learning_rate=2e-4, lr_scheduler_type='cosine', warmup_ratio=0.05,
            fp16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            logging_steps=10, save_strategy='epoch', save_total_limit=1,
            report_to='none', optim='paged_adamw_32bit',
            dataset_text_field='text', max_length=256,
        ),
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    final = os.path.join(out_dir, 'final')
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f'  Adapter saved: {final}')
    return final

def compute_ppl(model, tok, val_file=None, limit=200):
    if val_file is None:
        val_file = VAL_FILE
    model.eval()
    total_loss, total_n = 0.0, 0
    with open(val_file) as f:
        rows = [json.loads(l) for l in f][:limit]
    for ex in rows:
        text = ('### Instruction:\n' + ex['instruction']
                + '\n\n### Response:\n' + ex['output'])
        inp = tok(text, return_tensors='pt', max_length=512, truncation=True).to(model.device)
        with torch.no_grad():
            loss = model(**inp, labels=inp['input_ids']).loss.item()
        total_loss += loss * inp['input_ids'].shape[-1]
        total_n += inp['input_ids'].shape[-1]
    return math.exp(total_loss / total_n)

def gen(model, tok, question, max_new=256):
    prompt = (
        '<s>[INST] <<SYS>>\n' + SYSTEM + '\n<</SYS>>\n\n'
        + '### Instruction:\n' + question + '\n\n### Response:\n'
    )
    inp = tok(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, temperature=0.7, top_p=0.9,
            do_sample=True, pad_token_id=tok.eos_token_id
        )
    return tok.decode(out[0][inp['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

print('Helpers ready.')


# ## Baseline Training (r=16, full dataset)
# 
# ~80 min on T4. This is the primary adapter used in the qualitative comparison.
# Skip and load a pre-trained adapter if you already have one.

# In[11]:


model, tok = load_model(rank=16)
baseline_path = run_train(model, tok, LOCAL_CKPT + '/r16_full')
baseline_ppl = compute_ppl(model, tok)
print(f'\nFine-tuned perplexity (r=16, full data): {baseline_ppl:.2f}')
del model
torch.cuda.empty_cache()


# In[ ]:


import json as _j

_snap = {'baseline_ppl': round(baseline_ppl, 2), 'baseline_path': str(baseline_path)}
with open(DRIVE_RESULTS + '/baseline.json', 'w') as _f:
    _j.dump(_snap, _f, indent=2)
print('Saved → Drive:', DRIVE_RESULTS + '/baseline.json')


# ## Experiment 1: Rank Sweep (r=8 / r=16 / r=64)
# 
# Each run: 1000 examples, 3 epochs. Measures the quality-vs-parameter tradeoff.
# 
# Expected: r=16 gives ~95% of r=64 quality at 25% the adapter parameters.
# The perplexity gap between r=8 and r=64 should be small on this task.

# In[ ]:


rank_results = {}

for rank in [8, 16, 64]:
    print('\n' + '='*55)
    print(f'Rank r={rank}')
    m, t = load_model(rank=rank)
    path = run_train(m, t, LOCAL_CKPT + f'/r{rank}', n=1000)
    ppl = compute_ppl(m, t)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    rank_results[rank] = {'path': path, 'ppl': round(ppl, 2), 'params': n_params}
    print(f'r={rank}: ppl={ppl:.2f}, trainable={n_params:,}')
    del m
    torch.cuda.empty_cache()

print('\n' + '='*55)
print('RANK SWEEP RESULTS')
print(f'{"Rank":<8} {"Perplexity":<14} Trainable Params')
print('-' * 45)
for r in sorted(rank_results):
    v = rank_results[r]
    print(f'r={r:<6} {v["ppl"]:<14} {v["params"]:,}')


# In[ ]:


import json as _j

_serializable = {str(k): {'ppl': v['ppl'], 'params': v['params']} for k, v in rank_results.items()}
with open(DRIVE_RESULTS + '/rank_sweep.json', 'w') as _f:
    _j.dump(_serializable, _f, indent=2)
print('Saved → Drive:', DRIVE_RESULTS + '/rank_sweep.json')


# ## Experiment 2: Data Scaling (n=100 / n=500 / n=1000)
# 
# All runs: r=16, 3 epochs. Tests the data requirement threshold.
# 
# Expected: meaningful gains 100→500, diminishing returns beyond 500.
# Overfitting (train/val loss divergence) appears at epoch 3 for n=100.

# In[ ]:


scale_results = {}

for n in [100, 500, 1000]:
    print('\n' + '='*55)
    print(f'Data n={n}')
    m, t = load_model(rank=16)
    path = run_train(m, t, LOCAL_CKPT + f'/n{n}', n=n)
    ppl = compute_ppl(m, t)
    scale_results[n] = {'path': path, 'ppl': round(ppl, 2)}
    print(f'n={n}: ppl={ppl:.2f}')
    del m
    torch.cuda.empty_cache()

print('\n' + '='*55)
print('DATA SCALING RESULTS (r=16, 3 epochs)')
print(f'{"Examples":<12} Perplexity')
print('-' * 26)
for n in sorted(scale_results):
    print(f'{n:<12} {scale_results[n]["ppl"]}')


# In[ ]:


import json as _j

_serializable = {str(k): {'ppl': v['ppl']} for k, v in scale_results.items()}
with open(DRIVE_RESULTS + '/data_scaling.json', 'w') as _f:
    _j.dump(_serializable, _f, indent=2)
print('Saved → Drive:', DRIVE_RESULTS + '/data_scaling.json')


# ## Qualitative Comparison
# 
# Side-by-side base model vs fine-tuned. Fine-tuned answers should be more
# concise, technically precise, and follow the interview answer style.

# In[ ]:


EVAL_PROMPTS = [
    'What is LoRA and why is it more efficient than full fine-tuning?',
    'What is the difference between RAG and fine-tuning for LLMs?',
    'What is the vanishing gradient problem and how is it solved?',
]

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4'
)
base_m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map='auto', dtype=torch.float16, low_cpu_mem_usage=True)
base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
base_tok.pad_token = base_tok.eos_token

ft_m = PeftModel.from_pretrained(base_m, baseline_path)

for q in EVAL_PROMPTS:
    print('\n' + '='*70)
    print('Q:', q)
    print('\n[BASE MODEL]')
    print(gen(base_m, base_tok, q))
    print('\n[FINE-TUNED]')
    print(gen(ft_m, base_tok, q))

del base_m, ft_m
torch.cuda.empty_cache()


# ## Results Summary

# In[ ]:


import json as _j

print('=== RANK SWEEP (1000 examples, 3 epochs) ===')
print(f'{"Rank":<8} {"Perplexity":<14} Trainable Params')
print('-' * 45)
for r in sorted(rank_results):
    v = rank_results[r]
    note = ' <- default' if r == 16 else ''
    print(f'r={r:<6} {v["ppl"]:<14} {v["params"]:,}{note}')

print()
print('=== DATA SCALING (r=16, 3 epochs) ===')
print(f'{"Examples":<12} Perplexity')
print('-' * 26)
for n in sorted(scale_results):
    print(f'{n:<12} {scale_results[n]["ppl"]}')

print()
print(f'Baseline (r=16, full data): ppl={baseline_ppl:.2f}')

_summary = {
    'baseline': {'ppl': round(baseline_ppl, 2)},
    'rank_sweep': {str(k): {'ppl': v['ppl'], 'params': v['params']} for k, v in rank_results.items()},
    'data_scaling': {str(k): {'ppl': v['ppl']} for k, v in scale_results.items()},
}
with open(DRIVE_RESULTS + '/summary.json', 'w') as _f:
    _j.dump(_summary, _f, indent=2)
print()
print('Full summary saved → Drive:', DRIVE_RESULTS + '/summary.json')
print('Copy these numbers into lora-finetune/README.md under "LoRA Rank Experiments".')


# ## Save / Download Checkpoints
# 
# - `USE_DRIVE = True`: already saved to Drive, list paths below.
# - `USE_DRIVE = False`: download the baseline adapter as a zip file.

# In[ ]:


import zipfile

print('Results directory:', DRIVE_RESULTS)
for root, dirs, fnames in os.walk(DRIVE_RESULTS):
    for fname in fnames:
        fp = os.path.join(root, fname)
        print(f'  {fname:<40} {os.path.getsize(fp):>8,} bytes')

if IN_COLAB:
    zip_path = '/content/lora_finetune_results.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, fnames in os.walk(DRIVE_RESULTS):
            for fname in fnames:
                fp = os.path.join(root, fname)
                zf.write(fp, 'results/' + fname)
    print()
    print('Downloading lora_finetune_results.zip...')
    from google.colab import files
    files.download(zip_path)
else:
    print()
    print('Results at:', DRIVE_RESULTS)
    print('Sync back with: scp -r ubuntu@<EC2_IP>:' + DRIVE_RESULTS + ' ./results/')

print()
print('Copy results/*.json into lora-finetune/docs/ then update README.md.')


# ## Full vs Partial Results
# 
# Check `MyDrive/git_repos/lora-finetune/results/` in Google Drive — no runtime needed.
# 
# | Files present on Drive | Meaning |
# |---|---|
# | `summary.json` exists | **Full** — all experiments complete, safe to copy numbers |
# | `baseline.json` only | Disconnected after baseline, before rank sweep |
# | `baseline.json` + `rank_sweep.json` | Disconnected after rank sweep, before data scaling |
# | All three JSONs, no `summary.json` | Training done — re-run the Results Summary cell only |
# 
# The recovery cell below prints the same signal.

# ## Recovery — Re-Download Results After Reconnect
# 
# Session disconnected mid-run? Re-open the notebook, connect to **any** runtime (CPU is fine),
# and run only this cell. No retraining needed — everything is loaded from Drive.

# In[ ]:


import os, json, zipfile

try:
    from google.colab import drive, files as colab_files
    _in_colab = True
except ImportError:
    _in_colab = False

if _in_colab:
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    _results = '/content/drive/MyDrive/git_repos/lora-finetune/results'
else:
    # Local — reuse DRIVE_RESULTS from setup cell if available, else derive it
    try:
        _results = DRIVE_RESULTS
    except NameError:
        import pathlib
        _results = os.environ.get('RESULTS_DIR', str(pathlib.Path(__file__).resolve().parent)) + '/results'

def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None

baseline     = _load(_results + '/baseline.json')
rank_sweep   = _load(_results + '/rank_sweep.json')
data_scaling = _load(_results + '/data_scaling.json')

print('=== RECOVERED RESULTS ===')
print('Source:', _results)
if baseline:
    print(f'Baseline (r=16, full data) ppl: {baseline["baseline_ppl"]}')
if rank_sweep:
    print('\nRank sweep:')
    for r in sorted(rank_sweep, key=int):
        v = rank_sweep[r]
        print(f'  r={r}: ppl={v["ppl"]}  params={v.get("params","?")}')
if data_scaling:
    print('\nData scaling:')
    for n in sorted(data_scaling, key=int):
        print(f'  n={n}: ppl={data_scaling[n]["ppl"]}')
if not any([baseline, rank_sweep, data_scaling]):
    print('No results found — training may not have started.')

if _in_colab:
    zip_path = '/content/lora_finetune_recovery.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(_results):
            for root, dirs, fnames in os.walk(_results):
                for fname in fnames:
                    fp = os.path.join(root, fname)
                    zf.write(fp, 'results/' + fname)
    print('\nDownloading recovery zip...')
    colab_files.download(zip_path)
    print('Done.')
else:
    print('\nResults on disk at:', _results)
    print('Sync back with: scp -r ubuntu@<EC2_IP>:' + _results + ' ./results/')

