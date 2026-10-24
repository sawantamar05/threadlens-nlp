"""Generate shareable aggregate reports from measured results, without source comments."""

from __future__ import annotations

import csv
import io
import json

from threadlens.config import LABELS, settings


def generate_reports() -> dict:
    path = settings.reports / "experiments.json"
    report = json.loads(path.read_text())
    partition_counts = "/".join(
        str(report["splits"][name]["count"]) for name in ["train", "validation", "test"]
    )
    partition_groups = "/".join(
        str(len(report["splits"][name]["thread_ids"])) for name in ["train", "validation", "test"]
    )
    request_validation = report["splits"]["validation"]["label_counts"].get("request", 0)
    request_test = report["splits"]["test"]["label_counts"].get("request", 0)
    environment_path = settings.reports / "environment.json"
    environment = report.get("environment") or (
        json.loads(environment_path.read_text()) if environment_path.exists() else None
    )
    environment_description = (
        json.dumps(environment, sort_keys=True)
        if environment
        else "not recorded in an environment snapshot; inspect per-run runtime metadata"
    )
    rows = []
    for run in report["runs"]:
        rows.append(
            {
                "configuration": run["configuration"],
                "seed": run["seed"],
                "validation_macro_f1": run["validation"]["macro_f1"],
                "test_accuracy": run["test"]["accuracy"],
                "test_macro_f1": run["test"]["macro_f1"],
                "request_recall": run["test"]["per_label"]["request"]["recall"],
                "training_seconds": run["training_duration_seconds"],
                "warm_mean_ms": run["latency"]["warm_single_mean_ms"],
                "device": run["latency"]["device"],
                "provenance": "weak_label",
            }
        )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (settings.reports / "per_seed_metrics.csv").write_text(output.getvalue())
    lines = [
        "# Experiment report",
        "",
        "These are measured agreements with unreviewed local NLI weak labels. Independent human semantic accuracy is **unmeasured**. The teacher is not evaluated against its own labels as an independent baseline.",
        "",
        f"Dataset: {report['dataset']['comment_count']} comments / {report['dataset']['thread_count']} threads. Frozen train/validation/test: {partition_counts} comments and {partition_groups} disjoint threads. Dataset/label hash: `{report['dataset']['dataset_hash']}`.",
        "",
        "## Protocol",
        "",
        "Quality goals were committed before test evaluation in `configs/preregistration.json`. Seeds 42, 7, and 2024 vary training randomness on fixed partitions; all seeds are reported. TF-IDF is fitted only on training text. Original DistilBERT: 3 epochs, learning rate 2e-5, batch 16, maximum 256 tokens, dynamic padding, AdamW, 0.01 weight decay, 10% linear warmup. Each run retains the epoch with highest validation macro F1; ties retain the earlier epoch. Only validation metrics select the final configuration. Seed 42 is the predeclared representative checkpoint, rather than a selected best seed.",
        "",
        "Only seed 42 model artifacts are retained to conserve local disk; every seed keeps local row predictions, metadata, training curves, confusion counts and metrics. The model/tokenizer were saved, loaded with local-files-only, and checked for unchanged logits and fixed label mappings. MPS randomness is not guaranteed bitwise deterministic.",
        "",
        "## Every measured run",
        "",
        "| Configuration | Seed | Val macro F1 | Test accuracy | Test macro F1 | Request recall | Train s | Warm mean ms | Device |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['configuration']} | {row['seed']} | {row['validation_macro_f1']:.4f} | {row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} | {row['request_recall']:.4f} | {row['training_seconds']:.2f} | {row['warm_mean_ms']:.2f} | {row['device']} |"
        )
    lines += [
        "",
        "## Variation across seeds",
        "",
        "Population standard deviation describes these three runs, not a confidence interval or uncertainty from resampling discussions.",
        "",
        "| Configuration | Test accuracy mean ± std | Test macro F1 mean ± std | Mean validation macro F1 |",
        "|---|---:|---:|---:|",
    ]
    for summary in report["summaries"]:
        lines.append(
            f"| {summary['configuration']} | {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f} | {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} | {summary['validation_macro_f1_mean']:.4f} |"
        )
    improvement = report.get("improvement")
    if improvement:
        lines += [
            "",
            "## One bounded improvement",
            "",
            f"Declared `{improvement['declared_at']}` in `configs/improvement.json`: **{improvement['change']}**. {improvement['reason']}",
            "",
            f"Before: `{json.dumps(improvement['before'], sort_keys=True)}`.",
            "",
            f"After: `{json.dumps(improvement['after'], sort_keys=True)}`.",
            "",
            "The declaration was committed before improved test evaluation. No further search was performed. Both configurations use exactly the same test comments. An improvement hypothesis is retained even when measured performance does not improve.",
        ]
    lines += [
        "",
        "## Per-label metrics and confusion counts",
        "",
        "Representative seed 42 runs. Confusion rows are reference labels; columns are predictions in analysis/request/commentary order. Full per-seed arrays are in `reports/experiments.json`.",
    ]
    for run in report["runs"]:
        if run["seed"] != 42:
            continue
        lines += [
            "",
            f"### {run['configuration']}",
            "",
            "| Label | Precision | Recall | F1 | Support | Predicted count |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for label in LABELS:
            metrics = run["test"]["per_label"][label]
            lines.append(
                f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {int(metrics['support'])} | {run['test']['prediction_counts'][label]} |"
            )
        lines += ["", "```text", *[str(row) for row in run["test"]["confusion_matrix"]], "```"]
    selected_id = report.get("selected_model", {}).get("model_id")
    if selected_id:
        selected = next(run for run in report["runs"] if run["model_id"] == selected_id)
        matrix = selected["test"]["confusion_matrix"]
        confusions = sorted(
            ((matrix[i][j], LABELS[i], LABELS[j]) for i in range(3) for j in range(3) if i != j),
            reverse=True,
        )
        top = "; ".join(
            f"{reference} → {predicted}: {count}" for count, reference, predicted in confusions[:3]
        )
        lines += [
            "",
            "## Selection, goals, and error analysis",
            "",
            f"Selected local transformer: `{selected_id}`, by mean validation macro F1. Most frequent seed 42 test disagreements: {top}.",
            "",
            "These are disagreements with silver references, not confirmed semantic errors. No public raw-text examples are redistributed. Local `.local/predictions/*-test.jsonl` preserves IDs, full source text, reference/predicted labels and scores for review. Inspect mixed-purpose comments, rhetorical questions and unsupported numerical claims in blind review before drawing semantic conclusions.",
            "",
            f"Input truncation in selected run: `{json.dumps(selected['metadata']['truncation_counts'])}`. Complete source text remains stored. A request or supporting evidence after token 256 can be missed by the trained model, even though the windowed teacher considered the full comment.",
            "",
            f"Only {request_validation} validation and {request_test} test records are weak-labeled request; minority metrics are unstable. Class imbalance favors commentary. Accuracy must be considered alongside macro F1, per-label recall and prediction counts.",
            "",
            "| Preregistered goal | Target | Observed | Met |",
            "|---|---:|---:|---|",
        ]
        for key, value in report.get("quality_goals", {}).items():
            if "target" in value:
                lines.append(
                    f"| {key} | {value['target']} | {value['observed']:.4f} | {value['met']} |"
                )
        lines += [
            "",
            "Semantic quality remains unmeasured even when a silver-agreement or latency goal passes. Functional software checks are separate from these model-quality goals.",
        ]
    lines += [
        "",
        "## Latency and reproducibility",
        "",
        f"Warm latency uses up to 20 sequential single-comment requests after 3 warmups, including tokenization and model inference, excluding HTTP/UI overhead. MPS is synchronized around each timer. Models use dynamic input lengths up to 256 tokens. scikit-learn runs use CPU. Original project inspection snapshot (future run hardware may differ): `{environment_description}`. Complete package versions, exact base-model revision, artifact identifier, dataset hash, split reference, source commit, per-epoch losses and durations are embedded in each run's metadata.",
        "",
        "Run `threadlens collect`, `threadlens label --device mps`, `threadlens split`, then `threadlens experiments --stage initial --device mps`. Commit the generated `configs/improvement.json` before `threadlens experiments --stage improved --device mps`. Use `--device cpu` on other platforms. See the README for locked setup and the data card for source-edit limitations. `reports/per_seed_metrics.csv` and `reports/experiments.json` are shareable aggregate artifacts.",
        "",
    ]
    (settings.root / "docs/experiments.md").write_text("\n".join(lines))
    if selected_id:
        _model_card(report, selected)
    return {
        "report": str(path),
        "markdown": "docs/experiments.md",
        "csv": "reports/per_seed_metrics.csv",
    }


def _model_card(report: dict, selected: dict) -> None:
    metadata = selected["metadata"]
    config = metadata["training_config"]
    environment_path = settings.reports / "environment.json"
    environment = (
        metadata.get("runtime")
        or report.get("environment")
        or (json.loads(environment_path.read_text()) if environment_path.exists() else None)
    )
    environment_description = (
        json.dumps(environment, sort_keys=True) if environment else "runtime not recorded"
    )
    test_groups = len(report["splits"]["test"]["thread_ids"])
    test_requests = report["splits"]["test"]["label_counts"].get("request", 0)
    text = f"""# Model card

ThreadLens is an original local discourse-classification portfolio application by Amar Sawant, implemented with AI assistance. It predicts analysis/request/commentary; it does not verify factual truth, rate authors, or establish content quality.

## Selected artifact

- Identifier: `{metadata["artifact_identifier"]}`; selected configuration: `{selected["configuration"]}`; fixed representative seed: 42.
- Base: [`{config["base_model"]}`](https://huggingface.co/{config["base_model"]}), Apache-2.0, revision `{config["base_revision"]}`.
- Label mapping: analysis=0, request=1, commentary=2.
- Actual training device: `{metadata["device"]}`; environment: `{environment_description}`; CPU supported through the same training and inference code. Auto mode chooses MPS when available, otherwise CPU; model metadata and doctor output expose the actual choice. Explicit MPS mode fails clearly if unavailable.
- Source commit: `{metadata["code_commit"]}`. Dataset/label hash: `{metadata["dataset_hash"]}`.
- Training configuration: `{json.dumps(config, sort_keys=True)}`.
- Duration: {metadata["training_duration_seconds"]:.2f} seconds; chosen epoch: {metadata["selected_epoch"]}. Package versions: `{json.dumps(metadata["package_versions"], sort_keys=True)}`.

Weights, tokenizer, local database and raw comments are excluded from normal Git. Run the documented collection/labeling/splitting/training commands to rebuild. The artifact was saved with safetensors, reloaded locally, and checked for fixed label mapping and matching logits. The API caches each requested model artifact once per process; Streamlit calls the API and does not load a second copy.

## Measured agreement, not independent accuracy

All {report["dataset"]["comment_count"]} source labels are unreviewed `weak_label` outputs from a separate local NLI model; there are zero independent human benchmark labels. The selected seed 42 test agreement is accuracy={selected["test"]["accuracy"]:.4f}, macro F1={selected["test"]["macro_f1"]:.4f} on {selected["test"]["support"]} comments from {test_groups} held-out threads. Its warm mean single-comment model latency was {selected["latency"]["warm_single_mean_ms"]:.2f}ms over {selected["latency"]["sample_count"]} requests after 3 warmups, excluding network/UI overhead. See [all seeds and the bounded before/after experiment](experiments.md); seed 42 is not a best-seed selection.

Configuration selection uses mean validation macro F1 across seeds 42, 7, and 2024. Test data is held out of tuning. Same-thread comments never span partitions. Seeds measure optimization randomness on fixed groups and do not estimate split uncertainty. The test has only {test_requests} weak-labeled requests; minority-class estimates are unstable. The teacher is never treated as an independent baseline against labels it generated.

## Scores, truncation, and limitations

Scores are uncalibrated softmax outputs, not probabilities that a prediction is correct. A fixed 0.60 maximum-score threshold flags low confidence; it was not tuned on test data. Neural explanations are not invented. Majority and TF-IDF alternatives use their own documented score outputs.

Complete comments are stored locally. DistilBERT sees at most 256 tokens, and inference responses/UI explicitly flag truncation. Selected training/validation/test truncation counts: `{json.dumps(metadata["truncation_counts"])}`. A late request or supporting evidence may be missed. Mixed-purpose comments, rhetorical questions, quotation attribution, unsupported numbers, sarcasm, code snippets and context-dependent replies are difficult. The model sees a comment alone rather than its parent/reply context. Pretraining and the English-only HN collection introduce demographic, topical and linguistic biases. No independent human validation, production deployment, real users or business impact has been established.

The demo classifier is trained only on original synthetic fixtures and has `synthetic_demo` provenance. It is an offline software demonstration and is separate from this real-source weak-label benchmark. The annotation UI supports independent blind first labels and records revisions, enabling a future explicitly versioned human benchmark without relabeling this frozen experiment after inspecting errors.

Intended use is exploratory discussion analysis with human review. Do not use these labels as automated moderation decisions, factual verification, employee assessment, or evidence of a person's expertise. See [the data card](data_card.md) for source retention and taxonomy ambiguity and the API's `/models` response for active runtime metadata.
"""
    (settings.root / "docs/model_card.md").write_text(text)
