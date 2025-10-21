from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_company_column(df: pd.DataFrame) -> str:
    """Return the column name that stores company identifiers."""
    candidate_aliases = {
        "company",
        "company_name",
        "companyid",
        "company_id",
        "entity",
        "issuer",
    }
    lowered = {col.lower(): col for col in df.columns}
    for alias in candidate_aliases:
        if alias in lowered:
            return lowered[alias]
    for col in df.columns:
        if "company" in col.lower():
            return col
    raise ValueError(
        "Could not detect a company column. "
        f"Available columns: {', '.join(df.columns)}"
    )


def ensure_binary_labels(df: pd.DataFrame, label_cols: Iterable[str]) -> pd.DataFrame:
    """Ensure label columns contain binary {0,1} values."""
    label_cols = list(label_cols)
    missing = [col for col in label_cols if col not in df.columns]
    if missing:
        alias_map = {
            "social": ["soc", "soc_label", "social_label"],
            "environment": ["env", "env_label", "environmental", "environment_label"],
            "financial": ["fin", "fin_label", "financial_label"],
            "maori": ["maori_label"],
        }
        rename_map = {}
        for target in missing:
            candidates = alias_map.get(target, [])
            found = None
            for col in df.columns:
                normalized = col.lower().replace(" ", "").replace("-", "").replace("_", "")
                target_norm = target.lower().replace(" ", "").replace("-", "").replace("_", "")
                if normalized == target_norm:
                    found = col
                    break
                for candidate in candidates:
                    candidate_norm = (
                        candidate.lower().replace(" ", "").replace("-", "").replace("_", "")
                    )
                    if normalized == candidate_norm:
                        found = col
                        break
                if found:
                    break
            if found:
                rename_map[found] = target
            else:
                synonyms = {col.rstrip("_label").rstrip("_lbl") for col in df.columns}
                raise KeyError(
                    f"Missing label column '{target}'. "
                    f"Available columns: {', '.join(df.columns)}. "
                    f"Provide matching column names (synonyms include: {', '.join(sorted(synonyms))})."
                )
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

    for col in label_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        unique_values = set(df[col].unique().tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"Column {col} contains non-binary values: {unique_values}")
    return df


def safe_mkdirs(path: str | Path) -> Path:
    """Create directories safely and return the created Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def pretty_label_stats(df: pd.DataFrame, label_cols: Iterable[str]) -> None:
    """Print human-readable label counts and prevalences."""
    label_cols = list(label_cols)
    total = len(df)
    print(f"Total records: {total}")
    for col in label_cols:
        positives = int(df[col].sum())
        prevalence = positives / total if total else 0.0
        print(f"{col}: {positives} positives ({prevalence:.2%})")
