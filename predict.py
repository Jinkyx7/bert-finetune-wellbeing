from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import safe_mkdirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained multi-label classifier.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
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


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_path = Path(args.out_csv)
    safe_mkdirs(output_path.parent)

    df = pd.read_csv(args.csv_path)
    df[args.text_col] = df[args.text_col].astype(str).str.strip()
    df = df[df[args.text_col].astype(bool)].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dataset = prepare_dataset(df, args.text_col, args.label_cols)

    def collate(batch):
        texts = [item[args.text_col] for item in batch]
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return tokens

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    all_probs: List[np.ndarray] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    if all_probs:
        y_prob = np.concatenate(all_probs, axis=0)
    else:
        y_prob = np.zeros((len(df), len(args.label_cols)), dtype=float)

    thresholds = load_thresholds(model_dir, args.label_cols)
    y_pred = (y_prob >= np.array([thresholds[label] for label in args.label_cols])).astype(int)

    prob_columns = [f"prob_{label}" for label in args.label_cols]
    pred_columns = [f"pred_{label}" for label in args.label_cols]

    output_df = df[[args.text_col]].copy()
    for idx, label in enumerate(args.label_cols):
        output_df[prob_columns[idx]] = y_prob[:, idx]
        output_df[pred_columns[idx]] = y_pred[:, idx]

    output_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()

