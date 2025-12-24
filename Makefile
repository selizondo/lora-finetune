.PHONY: install prepare evaluate inference colab

install:
	pip install -r requirements.txt

# Download and prepare dataset (CPU, ~2 min)
prepare:
	python prepare_dataset.py

# Evaluate a trained adapter (requires checkpoints/final/ from Colab run)
evaluate:
	python evaluate.py --adapter ./checkpoints/final --mode both

# Interactive inference with a trained adapter
inference:
	python inference.py --adapter ./checkpoints/final --interactive

# GPU training requires Colab T4. This target prints instructions.
colab:
	@echo ""
	@echo "GPU training requires Google Colab (T4 runtime, free tier)."
	@echo ""
	@echo "  1. Open train_colab.ipynb in Colab"
	@echo "  2. Runtime → Change runtime type → T4 GPU"
	@echo "  3. Add HF_TOKEN to Colab Secrets (key icon, left sidebar)"
	@echo "  4. Run all cells — ~80 min total"
	@echo "  5. Results saved to Google Drive: MyDrive/git_repos/lora-finetune/results/"
	@echo ""
