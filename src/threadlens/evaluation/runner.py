"""Resumable bounded experiment stages; no test-directed hyperparameter search."""

from __future__ import annotations

import json
import subprocess
from collections import Counter

import numpy as np

from threadlens.config import LABELS, settings
from threadlens.data.hn import utc_now
from threadlens.data.splits import freeze_splits
from threadlens.evaluation.metrics import summarize_runs
from threadlens.models.training import read_labeled, train_baselines, train_transformer


def _report(
    records: list[dict], split: dict, runs: list[dict], improvement: dict | None = None
) -> dict:
    result = {
        "generated_at": utc_now(),
        "dataset": {
            "comment_count": len(records),
            "thread_count": len({r["thread_id"] for r in records}),
            "label_counts": dict(Counter(r["label"] for r in records)),
            "provenance": "weak_label",
            "dataset_hash": split["dataset_hash"],
        },
        "splits": {
            name: {key: value for key, value in part.items() if key != "indices"}
            for name, part in split["partitions"].items()
        },
        "runs": runs,
        "summaries": summarize_runs(runs),
        "improvement": improvement,
        "quality_status": "Provisional weak-label agreement; independent semantic accuracy unmeasured.",
        "benchmark_provenance": "weak_label",
        "independent_human_labels": 0,
        "teacher_not_evaluated_against_own_labels": True,
        "seed_interpretation": "Seeds42,7,2024 measure training randomness on exactly the same frozen story partitions, not split uncertainty.",
        "test_usage": "Test partition used only for final reporting; configuration selection uses mean validation macro F1.",
        "class_labels": list(LABELS),
        "preregistration": json.loads((settings.root / "configs/preregistration.json").read_text()),
    }
    path = settings.reports / "experiments.json"
    path.write_text(json.dumps(result, indent=2))
    return result


def declare_improvement(runs: list[dict]) -> dict:
    """Inspect validation/training evidence only, then freeze exactly one change."""
    path = settings.root / "configs/improvement.json"
    if path.exists():
        return json.loads(path.read_text())
    original = [run for run in runs if run["configuration"] == "distilbert-original"]
    mean_recalls = {
        label: float(np.mean([run["validation"]["per_label"][label]["recall"] for run in original]))
        for label in LABELS
    }
    worst_label = min(mean_recalls, key=mean_recalls.get)
    weighted = mean_recalls[worst_label] < 0.5
    config = {
        "declared_at": utc_now(),
        "declared_before_improved_test_evaluation": True,
        "change": "class-weighted cross-entropy" if weighted else "training budget: 3 to 5 epochs",
        "reason": f"Validation mean recall for {worst_label} was {mean_recalls[worst_label]:.4f}; train-only inverse-frequency weights address neglected classes."
        if weighted
        else "All validation mean class recalls were >=0.5; original learning curves justify testing one longer optimization budget without changing other hyperparameters.",
        "validation_mean_recalls": mean_recalls,
        "original_training_losses": {
            str(run["seed"]): [epoch["train_loss"] for epoch in run["epoch_history"]]
            for run in original
        },
        "before": {
            "epochs": 3,
            "class_weighted_loss": False,
            "learning_rate": 2e-5,
            "batch_size": 16,
            "max_length": 256,
        },
        "after": {
            "epochs": 3 if weighted else 5,
            "class_weighted_loss": weighted,
            "learning_rate": 2e-5,
            "batch_size": 16,
            "max_length": 256,
        },
        "selection_data": "train and validation only; no test metrics consulted by this decision function",
        "seeds": [42, 7, 2024],
        "further_search": "none",
    }
    path.write_text(json.dumps(config, indent=2))
    return config


def run_experiments(*, stage: str = "initial", device: str = "auto") -> dict:
    if stage not in {"initial", "improved", "all"}:
        raise ValueError("Stage must be initial, improved, or all.")
    records = read_labeled()
    split = freeze_splits(records)
    path = settings.reports / "experiments.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    if existing and existing["dataset"]["dataset_hash"] != split["dataset_hash"]:
        raise ValueError("Experiment report belongs to a different frozen dataset.")
    runs = existing.get("runs", [])
    if not any(run["configuration"] == "tfidf" for run in runs):
        runs.extend(train_baselines(records, split))
        _report(records, split, runs)
    for seed in [42, 7, 2024]:
        if not any(
            run["configuration"] == "distilbert-original" and run["seed"] == seed for run in runs
        ):
            runs.append(train_transformer(records, split, seed=seed, device=device))
            _report(records, split, runs)
    improvement = declare_improvement(runs)
    result = _report(records, split, runs, improvement)
    if stage == "initial":
        print(
            "Original experiments saved. Commit configs/improvement.json before running stage improved.",
            flush=True,
        )
        return result
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "configs/improvement.json"],
        cwd=settings.root,
        capture_output=True,
    )
    unchanged = subprocess.run(
        ["git", "diff", "HEAD", "--exit-code", "--", "configs/improvement.json"],
        cwd=settings.root,
        capture_output=True,
    )
    if tracked.returncode or unchanged.returncode:
        raise RuntimeError(
            "Commit the validation-derived configs/improvement.json before improved test evaluation. No user annotation is required."
        )
    for seed in [42, 7, 2024]:
        if not any(
            run["configuration"] == "distilbert-improved" and run["seed"] == seed for run in runs
        ):
            runs.append(
                train_transformer(
                    records,
                    split,
                    seed=seed,
                    epochs=improvement["after"]["epochs"],
                    weighted=improvement["after"]["class_weighted_loss"],
                    device=device,
                    configuration="distilbert-improved",
                )
            )
            _report(records, split, runs, improvement)
    result = _report(records, split, runs, improvement)
    transformer_summaries = [
        summary
        for summary in result["summaries"]
        if summary["configuration"].startswith("distilbert")
    ]
    selected = max(transformer_summaries, key=lambda summary: summary["validation_macro_f1_mean"])
    selected_id = selected["configuration"] + "-seed42"
    registry_path = settings.artifacts / "model_registry.json"
    registry = json.loads(registry_path.read_text())
    registry["active_model"] = selected_id
    registry_path.write_text(json.dumps(registry, indent=2))
    selected_run = next(run for run in runs if run["model_id"] == selected_id)
    goals = result["preregistration"]["goals"]
    result["selected_model"] = {
        "model_id": selected_id,
        "selection": "configuration by mean validation macro F1; seed42 fixed before experiments",
        "validation_macro_f1_mean": selected["validation_macro_f1_mean"],
    }
    result["quality_goals"] = {
        "silver_macro_f1": {
            "target": goals["silver_macro_f1"],
            "observed": selected["macro_f1_mean"],
            "met": selected["macro_f1_mean"] >= goals["silver_macro_f1"],
        },
        "silver_request_recall": {
            "target": goals["silver_request_recall"],
            "observed": selected_run["test"]["per_label"]["request"]["recall"],
            "met": selected_run["test"]["per_label"]["request"]["recall"]
            >= goals["silver_request_recall"],
        },
        "warm_single_latency_ms": {
            "target": goals["warm_single_latency_ms"],
            "observed": selected_run["latency"]["warm_single_mean_ms"],
            "met": selected_run["latency"]["warm_single_mean_ms"]
            <= goals["warm_single_latency_ms"],
        },
        "semantic_quality": {
            "status": "unmeasured",
            "reason": "No independent human benchmark; agreement with the NLI teacher is provisional.",
        },
    }
    result["complete"] = True
    path.write_text(json.dumps(result, indent=2))
    return result
