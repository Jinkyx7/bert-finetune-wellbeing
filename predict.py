from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import safe_mkdirs
from src.utils.pdf_processor import extract_sentences_with_pages, safe_report_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained multi-label classifier.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--pdf_path", type=str, default=None, help="Optional PDF file to extract sentences from.")
    parser.add_argument("--text_col", type=str, default="sentence")
    parser.add_argument(
        "--label_cols",
        nargs="+",
        default=["social", "environment", "financial", "maori"],
    )
    parser.add_argument("--out_csv", type=str, default="outputs/predictions.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def load_thresholds(model_dir: Path, label_cols: List[str]) -> Dict[str, float]:
    thresholds_path = model_dir / "thresholds.json"
    if thresholds_path.exists():
        with thresholds_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return {label: float(data.get(label, 0.5)) for label in label_cols}
    return {label: 0.5 for label in label_cols}


def prepare_dataset(df: pd.DataFrame, text_col: str, label_cols: List[str]) -> Dataset:
    base_columns = [text_col]
    for col in label_cols:
        if col in df.columns:
            base_columns.append(col)
    dataset = Dataset.from_pandas(df[base_columns], preserve_index=False)
    return dataset


def load_model_artifacts(model_dir: Path) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_dataframe(
    df: pd.DataFrame,
    *,
    text_col: str,
    label_cols: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    thresholds: Dict[str, float],
    batch_size: int = 32,
    max_length: int = 256,
) -> pd.DataFrame:
    working_df = df.copy()
    if text_col not in working_df.columns:
        raise KeyError(f"Column '{text_col}' not found in dataframe.")

    working_df[text_col] = working_df[text_col].astype(str).str.strip()
    working_df = working_df[working_df[text_col].astype(bool)].reset_index(drop=True)

    dataset = prepare_dataset(working_df, text_col, label_cols)

    def collate(batch):
        texts = [item[text_col] for item in batch]
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tokens

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    if all_probs:
        y_prob = np.concatenate(all_probs, axis=0)
    else:
        y_prob = np.zeros((len(working_df), len(label_cols)), dtype=float)

    y_pred = (y_prob >= np.array([thresholds[label] for label in label_cols])).astype(int)

    prob_columns = [f"prob_{label}" for label in label_cols]
    pred_columns = [f"pred_{label}" for label in label_cols]

    metadata_columns: List[str] = [text_col]
    for optional_col in ("source_page", "source_pdf"):
        if optional_col in working_df.columns:
            metadata_columns.append(optional_col)

    output_df = working_df[metadata_columns].copy()
    for idx, label in enumerate(label_cols):
        output_df[prob_columns[idx]] = y_prob[:, idx]
        output_df[pred_columns[idx]] = y_pred[:, idx]

    return output_df


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_path = Path(args.out_csv)
    safe_mkdirs(output_path.parent)

    if args.pdf_path:
        sentences = extract_sentences_with_pages(args.pdf_path)
        if not sentences:
            raise ValueError(f"No sentences extracted from PDF: {args.pdf_path}")
        df = pd.DataFrame(sentences)
        df.rename(columns={"sentence": args.text_col}, inplace=True)
        df["source_pdf"] = safe_report_name(args.pdf_path)
        df.rename(columns={"page": "source_page"}, inplace=True)
    else:
        if not args.csv_path:
            raise ValueError("Provide either --csv_path or --pdf_path for inference.")
        df = pd.read_csv(args.csv_path)

    tokenizer, model, device = load_model_artifacts(model_dir)
    thresholds = load_thresholds(model_dir, args.label_cols)
    output_df = predict_dataframe(
        df,
        text_col=args.text_col,
        label_cols=args.label_cols,
        tokenizer=tokenizer,
        model=model,
        device=device,
        thresholds=thresholds,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    output_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
