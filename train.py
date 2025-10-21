from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src.data import load_tokenized_datasets
from src.metrics import compute_metrics_factory, tune_thresholds
from src.split import split_within_company
from src.utils import safe_mkdirs, seed_everything


class WeightedTrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels.float())
        if return_outputs:
            return loss, outputs
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune XLM-Roberta for multi-label classification.")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--csv_path", type=str, default="dataset/combined_labels.csv")
    parser.add_argument("--text_col", type=str, default="sentence")
    parser.add_argument(
        "--label_cols",
        nargs="+",
        default=["social", "environment", "financial", "maori"],
    )
    parser.add_argument("--output_dir", type=str, default="outputs/xlmr-within-v1")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune_thresholds", action="store_true")
    return parser.parse_args()


def compute_class_weights(train_df, label_cols: List[str]) -> torch.Tensor:
    totals = len(train_df)
    weights = []
    for col in label_cols:
        positives = float(train_df[col].sum())
        negatives = totals - positives
        if positives == 0:
            print(f"Warning: label '{col}' has no positives in training data. Using weight = 1.0.")
            weights.append(1.0)
        else:
            weights.append(negatives / positives if positives else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def warn_missing_labels(split_name: str, df, label_cols: List[str]) -> None:
    for col in label_cols:
        unique = set(df[col].unique())
        if len(unique) <= 1:
            print(f"Warning: split '{split_name}' has constant label for '{col}': {unique}.")


def save_metrics(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump({k: float(v) for k, v in metrics.items()}, fp, indent=2)


def main() -> None:
    args = parse_args()
    if args.no_cuda:
        args.fp16 = False
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("Running on CPU (no CUDA/MPS).")

    seed_everything(args.seed)

    print("Creating within-company splits...")
    splits = split_within_company(
        csv_path=args.csv_path,
        text_col=args.text_col,
        label_cols=args.label_cols,
        company_col=None,
        seed=args.seed,
        ratios=(0.8, 0.1, 0.1),
    )

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    warn_missing_labels("validation", val_df, args.label_cols)
    warn_missing_labels("test", test_df, args.label_cols)

    tokenized_datasets, tokenizer = load_tokenized_datasets(
        model_name=args.model_name,
        data_dir="dataset",
        text_col=args.text_col,
        label_cols=args.label_cols,
        max_length=args.max_length,
    )

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(args.label_cols),
    )
    config.problem_type = "multi_label_classification"
    config.id2label = {idx: label for idx, label in enumerate(args.label_cols)}
    config.label2id = {label: idx for idx, label in enumerate(args.label_cols)}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
    )

    pos_weight = compute_class_weights(train_df, args.label_cols)
    print(f"Using class weights: {pos_weight.tolist()}")

    training_args = build_training_arguments(args)

    compute_metrics = compute_metrics_factory(args.label_cols)

    trainer = WeightedTrainer(
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    safe_mkdirs(args.output_dir)

    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    thresholds = None
    val_metrics = {}

    if len(val_dataset) > 0:
        print("Evaluating on validation set...")
        val_metrics = trainer.evaluate(eval_dataset=val_dataset)
        save_metrics(Path(args.output_dir) / "metrics_val.json", val_metrics)

        val_predictions = trainer.predict(val_dataset)
        logits_val = val_predictions.predictions
        y_true_val = val_predictions.label_ids

        if args.tune_thresholds:
            y_prob_val = 1.0 / (1.0 + np.exp(-logits_val))
            thresholds_path = Path(args.output_dir) / "thresholds.json"
            thresholds = tune_thresholds(
                y_true=y_true_val,
                y_prob=y_prob_val,
                label_cols=args.label_cols,
                output_path=thresholds_path,
            )
            print(f"Saved tuned thresholds to {thresholds_path}")
            tuned_metrics_fn = compute_metrics_factory(args.label_cols, thresholds=thresholds)
            tuned_metrics = tuned_metrics_fn(
                EvalPrediction(predictions=logits_val, label_ids=y_true_val)
            )
            save_metrics(Path(args.output_dir) / "metrics_val.json", tuned_metrics)
            val_metrics = tuned_metrics
    else:
        print("Validation dataset is empty. Skipping evaluation and threshold tuning.")

    if len(test_dataset) > 0:
        print("Evaluating on test set...")
        test_predictions = trainer.predict(test_dataset)
        logits_test = test_predictions.predictions
        y_true_test = test_predictions.label_ids

        metric_fn = compute_metrics_factory(args.label_cols, thresholds=thresholds)
        test_metrics = metric_fn(EvalPrediction(predictions=logits_test, label_ids=y_true_test))
        save_metrics(Path(args.output_dir) / "metrics_test.json", test_metrics)
        print("Validation metrics:", val_metrics)
        print("Test metrics:", test_metrics)
    else:
        print("Test dataset is empty. Skipping evaluation.")


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters)

    def maybe_set(kwargs: dict, key: str, value) -> None:
        if key in sig_params:
            kwargs[key] = value
        else:
            print(f"Warning: TrainingArguments does not support '{key}'. Skipping.")

    training_kwargs = {}
    maybe_set(training_kwargs, "output_dir", args.output_dir)
    maybe_set(training_kwargs, "learning_rate", args.lr)
    maybe_set(training_kwargs, "per_device_train_batch_size", args.batch_size)
    maybe_set(training_kwargs, "per_device_eval_batch_size", args.batch_size)
    maybe_set(training_kwargs, "num_train_epochs", args.epochs)
    maybe_set(training_kwargs, "weight_decay", args.weight_decay)
    maybe_set(training_kwargs, "save_total_limit", 2)
    maybe_set(training_kwargs, "fp16", args.fp16)
    maybe_set(training_kwargs, "no_cuda", args.no_cuda)
    maybe_set(training_kwargs, "seed", args.seed)
    maybe_set(training_kwargs, "logging_steps", 50)
    maybe_set(training_kwargs, "report_to", [])

    if "evaluation_strategy" in sig_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig_params:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        print("Warning: evaluation_strategy not supported; validation will run on demand only.")

    if "save_strategy" in sig_params:
        training_kwargs["save_strategy"] = "epoch"
    elif "save_steps" in sig_params:
        training_kwargs["save_steps"] = 0
        print("Warning: save_strategy not supported; disabling periodic checkpoint saves.")

    maybe_set(training_kwargs, "load_best_model_at_end", True)
    maybe_set(training_kwargs, "metric_for_best_model", "macro_f1")
    maybe_set(training_kwargs, "greater_is_better", True)

    return TrainingArguments(**training_kwargs)


if __name__ == "__main__":
    main()
