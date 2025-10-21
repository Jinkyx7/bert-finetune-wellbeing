from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .utils import (
    ensure_binary_labels,
    find_company_column,
    pretty_label_stats,
    safe_mkdirs,
)


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _simple_split_indices(
    n_samples: int, train_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    split_point = max(1, int(round(train_fraction * n_samples)))
    split_point = min(split_point, n_samples - 1) if n_samples > 1 else n_samples
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    return train_idx, test_idx


def _split_company(
    df_company: pd.DataFrame,
    label_cols: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
) -> SplitResult:
    n_rows = len(df_company)
    if n_rows < 5:
        return SplitResult(train=df_company, val=df_company.iloc[0:0], test=df_company.iloc[0:0])

    train_ratio, val_ratio, test_ratio = ratios
    temp_ratio = val_ratio + test_ratio

    label_matrix = df_company[label_cols].values
    try:
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=temp_ratio, random_state=seed
        )
        train_idx, temp_idx = next(splitter.split(label_matrix, label_matrix))
    except ValueError:
        train_fraction = train_ratio
        train_idx, temp_idx = _simple_split_indices(n_rows, train_fraction, seed)

    train_df = df_company.iloc[train_idx]
    temp_df = df_company.iloc[temp_idx]

    if len(temp_df) < 2 or test_ratio == 0:
        val_df = temp_df
        test_df = temp_df.iloc[0:0]
        return SplitResult(train=train_df, val=val_df, test=test_df)

    secondary_seed = seed + 1
    test_fraction = test_ratio / temp_ratio if temp_ratio else 0.5
    label_matrix_temp = temp_df[label_cols].values
    try:
        sub_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_fraction, random_state=secondary_seed
        )
        val_idx, test_idx = next(sub_splitter.split(label_matrix_temp, label_matrix_temp))
    except ValueError:
        val_fraction = val_ratio / temp_ratio if temp_ratio else 0.5
        val_idx, test_idx = _simple_split_indices(len(temp_df), val_fraction, secondary_seed)

    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    return SplitResult(train=train_df, val=val_df, test=test_df)


def split_within_company(
    csv_path: str,
    text_col: str,
    label_cols: Sequence[str],
    company_col: str | None = None,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found. Available: {df.columns.tolist()}")

    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].astype(bool)].reset_index(drop=True)

    if company_col is None:
        company_col = find_company_column(df)

    df = ensure_binary_labels(df, label_cols)

    train_frames: List[pd.DataFrame] = []
    val_frames: List[pd.DataFrame] = []
    test_frames: List[pd.DataFrame] = []

    rng = np.random.default_rng(seed)
    companies = df[company_col].unique()
    rng.shuffle(companies)

    for idx, company in enumerate(companies):
        company_df = df[df[company_col] == company].reset_index(drop=True)
        result = _split_company(
            company_df,
            label_cols=label_cols,
            ratios=ratios,
            seed=seed + idx,
        )
        train_frames.append(result.train)
        if not result.val.empty:
            val_frames.append(result.val)
        if not result.test.empty:
            test_frames.append(result.test)

    train_df = pd.concat(train_frames, ignore_index=True)
    val_df = pd.concat(val_frames, ignore_index=True) if val_frames else df.iloc[0:0]
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else df.iloc[0:0]

    print("Train set label distribution:")
    pretty_label_stats(train_df, label_cols)
    print("\nValidation set label distribution:")
    pretty_label_stats(val_df, label_cols)
    print("\nTest set label distribution:")
    pretty_label_stats(test_df, label_cols)

    destination = safe_mkdirs("dataset")
    train_path = destination / "train.csv"
    val_path = destination / "val.csv"
    test_path = destination / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {"train": train_df, "val": val_df, "test": test_df}
