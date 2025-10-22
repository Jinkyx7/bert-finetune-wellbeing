from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _clean_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Trim whitespace, coerce text to string, and drop empty rows."""
    df = df.copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].astype(bool)].reset_index(drop=True)
    return df


def _load_csv(path: Path, text_col: str, label_cols: Sequence[str]) -> pd.DataFrame:
    """Load a CSV if present; otherwise return an empty frame with expected columns."""
    if not path.exists():
        return pd.DataFrame(columns=[text_col, *label_cols])
    df = pd.read_csv(path)
    return _clean_dataframe(df, text_col)


def _build_dataset(df: pd.DataFrame, text_col: str, label_cols: Sequence[str]) -> Dataset:
    """Wrap a pandas DataFrame in a Hugging Face Dataset while preserving labels."""
    if df.empty:
        data_dict = {text_col: []}
        for col in label_cols:
            data_dict[col] = []
        return Dataset.from_dict(data_dict)
    columns = [text_col, *label_cols]
    return Dataset.from_pandas(df[columns], preserve_index=False)


def _tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_col: str,
    label_cols: Sequence[str],
    max_length: int,
) -> Dataset:
    """Tokenize text examples and attach label tensors for Trainer compatibility."""

    def tokenize(batch):
        # Tokenize a minibatch with padding/truncation to a consistent sequence length.
        tokens = tokenizer(
            batch[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        # Preserve multi-label targets for each example in the same order as label_cols.
        labels = [
            [float(batch[col][idx]) for col in label_cols]
            for idx in range(len(batch[text_col]))
        ]
        tokens["labels"] = labels
        return tokens

    remove_columns = list(dataset.column_names)
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=remove_columns,
        desc=f"Tokenizing {text_col}",
    )
    tensor_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized.column_names:
        tensor_columns.append("token_type_ids")
    tokenized.set_format(type="torch", columns=tensor_columns)
    return tokenized


def load_tokenized_datasets(
    model_name: str,
    data_dir: str,
    text_col: str,
    label_cols: Sequence[str],
    max_length: int = 256,
) -> DatasetDict:
    """Load train/validation/test CSV splits and return tokenized Dataset objects."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    paths = {
        "train": Path(data_dir) / "train.csv",
        "validation": Path(data_dir) / "val.csv",
        "test": Path(data_dir) / "test.csv",
    }
    base_datasets: Dict[str, Dataset] = {}
    for split, path in paths.items():
        df = _load_csv(path, text_col=text_col, label_cols=label_cols)
        base_datasets[split] = _build_dataset(df, text_col=text_col, label_cols=label_cols)

    tokenized = DatasetDict(
        {
            split: _tokenize_dataset(
                dataset,
                tokenizer=tokenizer,
                text_col=text_col,
                label_cols=label_cols,
                max_length=max_length,
            )
            for split, dataset in base_datasets.items()
        }
    )
    return tokenized, tokenizer
