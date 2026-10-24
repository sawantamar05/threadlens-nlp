"""Explicit three-class metrics, including missing prediction classes."""

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from threadlens.config import LABELS


def compute_metrics(reference: list[str], predictions: list[str]) -> dict:
    if not reference or len(reference) != len(predictions):
        raise ValueError("Metrics need equal non-empty reference and prediction sequences.")
    if not set(reference + predictions) <= set(LABELS):
        raise ValueError("Unknown taxonomy label.")
    report = classification_report(
        reference, predictions, labels=list(LABELS), output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(reference, predictions)),
        "macro_f1": float(
            f1_score(reference, predictions, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "per_label": {label: report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(reference, predictions, labels=list(LABELS)).tolist(),
        "labels": list(LABELS),
        "support": len(reference),
        "prediction_counts": {label: Counter(predictions)[label] for label in LABELS},
    }


def summarize_runs(runs: list[dict]) -> list[dict]:
    summaries = []
    for configuration in dict.fromkeys(run["configuration"] for run in runs):
        selected = [run for run in runs if run["configuration"] == configuration]
        summary = {"configuration": configuration, "seeds": [r["seed"] for r in selected]}
        for metric in ["accuracy", "macro_f1"]:
            values = [r["test"][metric] for r in selected]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
        summary["validation_macro_f1_mean"] = float(
            np.mean([r["validation"]["macro_f1"] for r in selected])
        )
        summaries.append(summary)
    return summaries
