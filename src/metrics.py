from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from transformers import EvalPrediction


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid helper for converting logits to probabilities."""
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics_factory(
    label_cols: Sequence[str],
    thresholds: Dict[str, float] | None = None,
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """Build a metrics callback that applies label-wise thresholds and aggregates scores."""
    label_cols = list(label_cols)
    thresholds = thresholds or {label: 0.5 for label in label_cols}
    thresholds = {label: thresholds.get(label, 0.5) for label in label_cols}

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute F1 and AUROC metrics for multi-label predictions."""
        logits = eval_pred.predictions
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        y_prob = _sigmoid(np.asarray(logits))
        y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
        y_true = np.asarray(eval_pred.label_ids)
        if y_true.size == 0:
            empty_metrics = {
                "macro_f1": float("nan"),
                "micro_f1": float("nan"),
                "macro_auroc": float("nan"),
            }
            for label in label_cols:
                empty_metrics[f"f1_{label}"] = float("nan")
                empty_metrics[f"auroc_{label}"] = float("nan")
            empty_metrics["macro_f1_labels"] = float("nan")
            return empty_metrics
        preds = np.zeros_like(y_prob, dtype=int)
        for idx, label in enumerate(label_cols):
            preds[:, idx] = (y_prob[:, idx] >= thresholds[label]).astype(int)

        metrics: Dict[str, float] = {}
        metrics["macro_f1"] = float(
            f1_score(y_true, preds, average="macro", zero_division=0)
        )
        metrics["micro_f1"] = float(
            f1_score(y_true, preds, average="micro", zero_division=0)
        )
        per_label_f1 = []
        auroc_scores = []
        for idx, label in enumerate(label_cols):
            label_true = y_true[:, idx]
            label_pred = preds[:, idx]
            per_label_score = f1_score(label_true, label_pred, zero_division=0)
            metrics[f"f1_{label}"] = float(per_label_score)
            per_label_f1.append(per_label_score)
            unique = np.unique(label_true)
            if unique.size > 1:
                try:
                    score = roc_auc_score(label_true, y_prob[:, idx])
                    auroc_scores.append(score)
                    metrics[f"auroc_{label}"] = float(score)
                except ValueError as err:
                    print(f"Warning: AUROC failed for label '{label}' with error: {err}")
                    metrics[f"auroc_{label}"] = float("nan")
            else:
                metrics[f"auroc_{label}"] = float("nan")
        if auroc_scores:
            metrics["macro_auroc"] = float(np.mean(auroc_scores))
        else:
            metrics["macro_auroc"] = float("nan")
        metrics["macro_f1_labels"] = float(np.mean(per_label_f1))
        return metrics

    return compute_metrics


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_cols: Iterable[str],
    output_path: str | Path = "thresholds.json",
) -> Dict[str, float]:
    """Grid-search label-wise decision thresholds that maximize F1 scores."""
    label_cols = list(label_cols)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    thresholds = {}
    candidate_thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)

    for idx, label in enumerate(label_cols):
        best_threshold = 0.5
        best_score = -1.0
        label_true = y_true[:, idx]
        if np.unique(label_true).size <= 1:
            thresholds[label] = best_threshold
            continue
        label_probs = y_prob[:, idx]
        for threshold in candidate_thresholds:
            # Evaluate candidate thresholds by binary F1 score to select the best cutoff.
            preds = (label_probs >= threshold).astype(int)
            score = f1_score(label_true, preds, zero_division=0)
            if score > best_score or (np.isclose(score, best_score) and threshold < best_threshold):
                best_score = score
                best_threshold = threshold
        thresholds[label] = float(best_threshold)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    return thresholds
