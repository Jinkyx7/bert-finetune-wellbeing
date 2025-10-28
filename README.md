# XLM-R Wellbeing Classifier

## Environment Setup
Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Placement
Create `dataset/combined_labels.csv` by copying your labelled sentences into that exact path. The file is ignored by git by default, so every contributor must source it locally (or download from your agreed storage bucket) before training.

## Training Workflow
The training entry point performs an 80/10/10 within-company split, tokenizes sentences, and fine-tunes `xlm-roberta-base` with class-weighted BCE loss. All outputs are stored under the directory provided by `--output_dir` (default `outputs/xlmr-within-v1`), including checkpoints, metrics, and tuned thresholds. Append `--no_cuda` to force CPU execution when GPUs are unavailable.

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
`split_within_company` keeps every company represented in each split: sentences are stratified inside a company into 80% train, 10% validation, and 10% test. Companies with fewer than five sentences are kept entirely in the training set. Label distributions for each split are printed to help track class balance.

## Outputs and Metrics
During training the best checkpoint (macro-F1 on validation) is saved in `outputs/xlmr-within-v1`. Validation metrics are written to `metrics_val.json` (with tuned thresholds if enabled), the test evaluation to `metrics_test.json`, and optimal per-label thresholds to `thresholds.json`.

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

To analyse a PDF annual report directly, provide a `--pdf_path` instead of `--csv_path`:

```bash
python predict.py \
  --model_dir outputs/xlmr-within-v1 \
  --pdf_path reports/AIR2024.pdf \
  --out_csv outputs/air2024_sentences.csv
```

The PDF pipeline mirrors the `bert-document-scan` utilities: text is cleaned to fix common encoding issues, segmented into sentences, and filtered to retain informative spans between 30 and 600 characters. The resulting CSV includes the extracted sentence, `source_page`, and the canonical `source_pdf` identifier alongside probability and prediction columns.

For batch processing, use the helper script to scan an entire directory (defaults to `reports/`). Each PDF produces two reports: `<name>_predictions.csv` (probs+binary flags) and `<name>_probabilities.csv` (probs only):

```bash
python pdf_predict.py \
  --model_dir outputs/xlmr-within-v1 \
  --output_dir outputs/pdf_reports
```

You can still target specific files by listing them after the arguments (e.g., `python pdf_predict.py --model_dir ... reports/AIR2024.pdf`).***
