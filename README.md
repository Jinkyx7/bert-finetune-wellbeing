# XLM-R Wellbeing Classifier

> Fine-tune an XLM-R classifier for wellbeing labels and use the trained model for CSV/PDF inference.

## Environment Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project is built entirely on the PyTorch + Hugging Face stack:

- `transformers` powers the `Trainer`, `AutoModelForSequenceClassification`, and XLM-R backbones.
- `datasets` handles in-memory dataset objects for tokenization and efficient batching.
- `iterstrat` keeps train/val/test splits multilabel-stratified inside each company.
- `scikit-learn` provides F1 and AUROC metrics, while `numpy/pandas` drive preprocessing utilities.
- PDF inference supports optional OCR for scanned documents (requires Tesseract when enabled).

## Data Placement

Create `dataset/combined_labels.csv` by copying your labelled sentences into that exact path. The file is ignored by git by default, so every contributor must source it locally (or download from your agreed storage bucket) before training. The training script materializes `dataset/train.csv`, `dataset/val.csv`, and `dataset/test.csv` every time it runs so that tokenization uses the exact same records as the split step.

## Training Workflow

`train.py` is the canonical entry point and wires together the entire end-to-end pipeline:

1. **Company-aware splits.** `split_within_company` loads `dataset/combined_labels.csv`, infers the company column, normalizes label columns to binary, and uses `MultilabelStratifiedShuffleSplit` (via `iterstrat`) to produce 80/10/10 train/val/test splits _within each company_. Companies with fewer than five labeled sentences are kept wholly in the training set, preventing cross-company leakage.
2. **Tokenization.** `src/data.load_tokenized_datasets` wraps each CSV split in a Hugging Face `Dataset`, cleans empty rows, and tokenizes text with the requested backbone (default `xlm-roberta-base`) using max-length padding/truncation so the `Trainer` receives PyTorch tensors.
3. **Model construction.** `AutoConfig` sets `problem_type="multi_label_classification"` with label/name mappings, and `AutoModelForSequenceClassification` initializes the XLM-R encoder/head.
4. **Optimization & loss.** `WeightedTrainer` subclasses `transformers.Trainer` to replace the default loss with `torch.nn.BCEWithLogitsLoss(pos_weight=…)`, where `pos_weight` equals the inverse prevalence of each label in the training split. This directly counteracts class imbalance without oversampling.
5. **Training loop.** Standard `TrainingArguments` drive the Hugging Face training loop (evaluation/checkpoints per epoch, optional `--fp16`, deterministic seeds, etc.). Metrics are computed through `compute_metrics_factory`, which applies configurable label-wise thresholds, then reports macro/micro F1, per-label F1, and AUROC scores.
6. **Threshold tuning (optional).** When `--tune_thresholds` is provided, validation logits are converted to probabilities and a simple grid search (`0.05 → 0.95`) picks label-specific decision thresholds that maximize F1. These are saved to `thresholds.json` and reused for both validation summaries and held-out test evaluation.

All outputs are stored under the directory provided by `--output_dir` (default `outputs/xlmr-within-v1`), including checkpoints, tokenizer files, metrics, and tuned thresholds. Append `--no_cuda` to force CPU execution when GPUs are unavailable.

Example command:

```bash
python train.py \
  --model_name xlm-roberta-base \
  --csv_path dataset/combined_labels.csv \
  --text_col sentence \
  --label_cols social environment financial maori \
  --output_dir outputs/xlmr-within-v1 \
  --max_length 256 \
  --lr 2e-5 \
  --batch_size 16 \
  --epochs 3 \
  --fp16 \
  --tune_thresholds
```

For a lightweight CPU-only smoke test you can run:

```bash
python3 train.py --no_cuda --batch_size 16 --max_length 256
```

This uses the Python 3 interpreter explicitly, disables GPU usage, keeps the default dataset path and other hyperparameters, processes sequences up to 256 tokens, and trains with batches of 16 examples.

## Data Splitting

`split_within_company` keeps every company represented in each split: sentences are stratified inside a company into 80% train, 10% validation, and 10% test. Companies with fewer than five sentences are kept entirely in the training set. Label distributions for each split are printed to help track class balance, and the resulting splits are written to `dataset/train.csv`, `dataset/val.csv`, and `dataset/test.csv` for transparency.

## Outputs and Metrics

During training the best checkpoint (macro-F1 on validation) is saved in `outputs/xlmr-within-v1`. Validation metrics are written to `metrics_val.json` (with tuned thresholds if enabled), the test evaluation to `metrics_test.json`, and optimal per-label thresholds to `thresholds.json`. Metrics include macro/micro F1, per-label F1, macro AUROC, and per-label AUROC (when both label states are present).

## Inference

Run predictions on any CSV using the trained directory:

```bash
python predict.py \
  --model_dir outputs/xlmr-within-v1 \
  --csv_path dataset/combined_labels.csv \
  --text_col sentence \
  --label_cols social environment financial maori \
  --out_csv outputs/predictions.csv
```

The script loads saved thresholds when available and writes per-label probabilities (`prob_*`) and binary predictions (`pred_*`). Keep `--seed` at its default (42) for reproducible splits and results.

To analyse a PDF annual report directly, provide a `--pdf_path` instead of `--csv_path` (add `--enable_ocr` for scanned PDFs, or `--force_ocr` to OCR every page):

```bash
python predict.py \
  --model_dir outputs/xlmr-wellbeing \
  --pdf_path reports/AIR2024.pdf \
  --enable_ocr \
  --out_csv outputs/air2024_sentences.csv
```

The PDF pipeline mirrors the `bert-document-scan` utilities: text is cleaned to fix common encoding issues, segmented into sentences, and filtered to retain informative spans between 30 and 600 characters. The resulting CSV includes the extracted sentence, `source_page`, and the canonical `source_pdf` identifier alongside probability and prediction columns. For scanned PDFs, enable OCR with `--enable_ocr` (requires `pillow`, `pytesseract`, and a system Tesseract install). Use `--force_ocr` to OCR every page regardless of embedded text.

Install Tesseract (examples):

```bash
# macOS (Homebrew)
brew install tesseract

# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

For batch processing, use `predict.py` to scan an entire directory (defaults to `reports/`). Each PDF produces `<name>_predictions.csv` (probs+binary flags). Add `--save_probabilities` to also write `<name>_probabilities.csv`. Add `--enable_ocr` for scanned PDFs or `--force_ocr` to always OCR:

```bash
python predict.py \
  --model_dir outputs/xlmr-wellbeing \
  --reports_dir reports \
  --enable_ocr \
  --force_ocr \
  --out_dir outputs/pdf_reports \
  --save_probabilities
```

You can still target specific files by listing them with `--pdf_paths` (e.g., `python predict.py --model_dir ... --pdf_paths reports/AIR2024.pdf`).\*\*\*
