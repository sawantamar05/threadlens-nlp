"""Resumable local NLI weak labels; never represented as human annotations."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from threadlens.config import LABELS, settings
from threadlens.data.hn import utc_now

TEACHER = "cross-encoder/nli-distilroberta-base"
TEMPLATE_VERSION = "threadlens-taxonomy-v1"
HYPOTHESES = {
    "analysis": "The author's primary contribution is a substantive claim supported by a concrete observation, comparison, measurement, relevant quotation, or explanatory mechanism.",
    "request": "The author's primary contribution is a genuine request for other participants to provide information, clarification, advice, validation, or action.",
    "commentary": "The author's primary contribution is an opinion, reaction, joke, unsupported assertion, or resource share without a genuine request or a supported claim.",
}


def choose_device(requested: str = "auto") -> str:
    import torch

    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; choose --device cpu explicitly.")
    if requested not in {"mps", "cpu"}:
        raise ValueError("Device must be auto, mps, or cpu.")
    return requested


def label_dataset(*, device: str = "auto", batch_size: int = 12) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    raw = settings.root / "data/raw/comments.jsonl"
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    output = settings.local / "labeled.jsonl"
    existing = (
        [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    )
    completed = {r["comment_id"]: r for r in existing}
    config_path = settings.local / "labeler_config.json"
    source = json.loads((settings.root / "configs/model_sources.json").read_text())["teacher"]
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = {
            "model_id": TEACHER,
            "revision": source["revision"],
            "taxonomy_version": TEMPLATE_VERSION,
            "hypothesis_template": "direct NLI premise/hypothesis",
            "hypotheses": HYPOTHESES,
            "max_length": 512,
            "premise_window_tokens": 384,
            "overlap_tokens": 32,
            "window_aggregation": "mean entailment logits, softmax across taxonomy hypotheses",
            "created_at": utc_now(),
            "license": "apache-2.0",
            "provenance": "weak_label",
        }
        config_path.write_text(json.dumps(config, indent=2))
    if config["model_id"] != source["model_id"] or config["revision"] != source["revision"]:
        raise ValueError(
            "Cached labeler differs from the pinned public model configuration; use a new dataset version."
        )
    selected_device = choose_device(device)
    print(f"Weak labeler device: {selected_device}; revision {config['revision']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER, revision=config["revision"])
    model = (
        AutoModelForSequenceClassification.from_pretrained(TEACHER, revision=config["revision"])
        .to(selected_device)
        .eval()
    )
    mapping = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    entailment = next((index for label, index in mapping.items() if "entail" in label), None)
    # Canonical model card explicitly documents contradiction0, entailment1, neutral2.
    if entailment is None and model.config.num_labels == 3:
        entailment = 1
    if entailment is None:
        raise ValueError("Cannot identify teacher entailment mapping.")
    for index, record in enumerate(records):
        if record["comment_id"] in completed:
            if completed[record["comment_id"]]["text_hash"] != record["text_hash"]:
                raise ValueError(
                    "Source text changed for an already weak-labeled item; version the dataset."
                )
            continue
        tokens = tokenizer.encode(record["text"], add_special_tokens=False)
        windows = [tokens[start : start + 384] for start in range(0, len(tokens), 352)]
        premises = [tokenizer.decode(window, skip_special_tokens=True) for window in windows]
        pairs = [(premise, HYPOTHESES[label]) for premise in premises for label in LABELS]
        logits = []
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                part = pairs[start : start + batch_size]
                encoded = tokenizer(
                    [p[0] for p in part],
                    [p[1] for p in part],
                    padding=True,
                    truncation="only_first",
                    max_length=512,
                    return_tensors="pt",
                ).to(selected_device)
                logits.extend(model(**encoded).logits[:, entailment].cpu().tolist())
        values = np.array(logits).reshape(len(windows), len(LABELS)).mean(axis=0)
        values = np.exp(values - values.max())
        scores = values / values.sum()
        label = LABELS[int(scores.argmax())]
        labeled = {
            **record,
            "label": label,
            "scores": dict(zip(LABELS, scores.tolist(), strict=True)),
            "provenance": "weak_label",
            "review_status": "unreviewed",
            "labeler": {
                **config,
                "timestamp": utc_now(),
                "device": selected_device,
                "window_count": len(windows),
                "source_tokens": len(tokens),
            },
        }
        completed[record["comment_id"]] = labeled
        with output.open("a") as handle:
            handle.write(json.dumps(labeled, ensure_ascii=False) + "\n")
        if (index + 1) % 20 == 0:
            print(
                f"Weak labels {index + 1}/{len(records)}: {dict(Counter(r['label'] for r in completed.values()))}",
                flush=True,
            )
    result = {
        "count": len(completed),
        "label_counts": dict(Counter(r["label"] for r in completed.values())),
        "provenance": "weak_label",
        "review_status": "unreviewed",
        "labeler": config,
        "independent_semantic_validation": "unmeasured",
        "device": selected_device,
    }
    (settings.reports / "labeling_manifest.json").write_text(json.dumps(result, indent=2))
    if len(result["label_counts"]) < 3:
        raise RuntimeError(
            f"Teacher class collapse: {result['label_counts']}; do not train a misleading three-class benchmark."
        )
    return result
