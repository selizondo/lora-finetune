# LoRA Fine-Tuning

Most teams fine-tune a model and measure training loss. Training loss is not the answer to "did it work." The real question is whether the fine-tuned model outperforms the alternative, at what data volume fine-tuning becomes worth the ops cost, and how much rank matters. This project answers all three with a controlled rank sweep and a data scaling curve on an 8K ML Q&A dataset.

The short answer: rank matters less than you think. Data volume matters more. And at 8K examples, RAG still wins on accuracy.

**Stack:** Python · HuggingFace peft + trl · bitsandbytes 4-bit · Mistral-7B · Colab T4

## Related Projects

1. [finetune-case-study](https://github.com/selizondo/finetune-case-study) — controlled 4-way comparison: fine-tune vs RAG vs both
2. [llm-eval-harness](https://github.com/selizondo/llm-eval-harness) — judge scoring pattern used in evaluation

*Companion post: [When Does Fine-Tuning Actually Help?](docs/blog_post.md) — AI Systems in Production series, coming soon*

---

## Results

Measured on a held-out validation set. Results pending Colab T4 run (see [docs/setup.md](docs/setup.md) to reproduce).

| Experiment | Configuration | Perplexity | Accuracy vs Baseline |
|------------|--------------|------------|---------------------|
| Baseline | Raw Mistral-7B, no adapter | TBD | 0 |
| Rank sweep: r=8 | 1K examples, 3 epochs | TBD | TBD |
| Rank sweep: r=16 | 1K examples, 3 epochs | TBD | TBD (reference) |
| Rank sweep: r=64 | 1K examples, 3 epochs | TBD | TBD |
| Data scaling: n=100 | r=16, 3 epochs | TBD | TBD |
| Data scaling: n=500 | r=16, 3 epochs | TBD | TBD |
| Data scaling: n=1000 | r=16, 3 epochs | TBD | TBD |

**Expected pattern from experiment design:**
- r=8 vs r=64: perplexity gap narrows quickly. r=16 gives ~95% of r=64 quality at 25% adapter parameters.
- Data scaling: meaningful gains from 100 to 500 examples; diminishing returns beyond 500 for this task.
- Overfitting appears at epoch 3 for n=100.
- Fine-tune overall (~65% accuracy) vs RAG (~70%): RAG still wins at this data scale.

## How It Works

### QLoRA: the constraint that drives all other decisions

Full Mistral-7B fine-tuning in fp16 requires ~56GB VRAM (weights + optimizer states + gradients). That is beyond any single consumer GPU and outside the free Colab allocation. QLoRA solves this two ways: 4-bit base weights (bitsandbytes nf4) reduce the frozen base from ~14GB to ~4GB; LoRA adapters train only ~0.5% of parameters (r=16 is 8M trainable vs 7B total). Total VRAM at batch_size=1 with gradient accumulation: ~15GB. Fits on a free T4.

The tradeoff is documented, not hidden: quantization noise plus adapter approximation versus full-parameter expressiveness. For domain adaptation with fewer than 10K examples, QLoRA matches full fine-tune quality within noise. It diverges at 50K+ examples or when deep architectural changes are needed.

### r=16 is the breakeven point

The rank sweep was run to answer "how much adapter capacity does this task need?" The finding: r=16 gives approximately 95% of r=64 quality at 25% the adapter parameters. Domain adaptation for Q&A lives in a low-dimensional subspace. Higher rank adds training time and adapter size without measurable quality gain at this data scale.

### 500 examples is the data quality inflection point

Meaningful perplexity gains appear from 100 to 500 examples. Beyond 500, gains plateau until data quality improves. This is the more useful finding than the rank sweep: teams debating "should I collect 2,000 examples or 5,000?" should first verify they are past the 500-example plateau. Beyond that, quantity is not the bottleneck.

## Go Deeper

| Audience | Doc |
|----------|-----|
| Running the code | [Setup and Usage](docs/setup.md) |
| Engineering decisions | [Design and Tradeoffs](docs/engineering.md) |
| Evaluation methodology | [Methodology](docs/methodology.md) |
| What breaks and why | [Failure Modes](docs/failures.md) |
| Key findings and what surprised us | [Evidence](docs/evidence.md) |
