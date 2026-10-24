"""Train isolated original models on frozen thread partitions and reload real artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import random
import subprocess
import time

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from threadlens.config import ID2LABEL, LABEL2ID, LABELS, settings
from threadlens.data.hn import utc_now
from threadlens.evaluation.metrics import compute_metrics
from threadlens.labeling.weak import choose_device

BASE_MODEL = "distilbert/distilbert-base-uncased"


def read_labeled() -> list[dict]:
    return [
        json.loads(line) for line in (settings.local / "labeled.jsonl").read_text().splitlines()
    ]


def versions() -> dict:
    result = {}
    for name in [
        "torch",
        "transformers",
        "scikit-learn",
        "numpy",
        "pandas",
        "tokenizers",
        "huggingface-hub",
    ]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed (optional training dependency)"
    return result


def code_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=settings.root, text=True
    ).strip()


def partitions(records: list[dict], split: dict) -> dict:
    return {
        name: [records[i] for i in part["indices"]] for name, part in split["partitions"].items()
    }


def registry_entry(model_id: str, kind: str, metadata: dict) -> dict:
    path = f"{model_id}/model.joblib" if kind == "sklearn" else model_id
    return {
        "id": model_id,
        "name": metadata.get("display_name", model_id),
        "kind": kind,
        "path": path,
        "version": metadata["artifact_identifier"],
        "provenance": "weak_label",
        "metadata": metadata,
    }


def update_registry(entry: dict) -> None:
    settings.artifacts.mkdir(parents=True, exist_ok=True)
    path = settings.artifacts / "model_registry.json"
    registry = (
        json.loads(path.read_text())
        if path.exists()
        else {"active_model": entry["id"], "models": []}
    )
    registry["models"] = [model for model in registry["models"] if model["id"] != entry["id"]] + [
        entry
    ]
    path.write_text(json.dumps(registry, indent=2))


def base_metadata(model_id: str, split: dict, device: str, config: dict) -> dict:
    return {
        "model_id": model_id,
        "artifact_identifier": f"{model_id}-{split['dataset_hash'][:12]}",
        "created_at": utc_now(),
        "provenance": "weak_label",
        "evaluation_provenance": "weak_label",
        "semantic_quality": "unmeasured without independently human-labeled benchmark",
        "dataset_hash": split["dataset_hash"],
        "split_manifest": "reports/split_manifest.json",
        "code_commit": code_revision(),
        "package_versions": versions(),
        "device": device,
        "training_config": config,
        "max_length": config.get("max_length"),
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "score_semantics": "Uncalibrated model scores, not probabilities of correctness.",
    }


def save_predictions(
    model_id: str,
    partition: str,
    records: list[dict],
    predictions: list[str],
    scores: list[list[float]] | None = None,
) -> None:
    folder = settings.local / "predictions"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (record, predicted) in enumerate(zip(records, predictions, strict=True)):
        rows.append(
            {
                "comment_id": record["comment_id"],
                "thread_id": record["thread_id"],
                "reference_label": record["label"],
                "predicted_label": predicted,
                "provenance": record["provenance"],
                "scores": scores[index] if scores else None,
                "text": record["text"],
            }
        )
    (folder / f"{model_id}-{partition}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def measure_latency(predict, texts: list[str], device: str) -> dict:
    if device == "mps":
        import torch

    sample = texts[: min(20, len(texts))]
    for text in sample[:3]:
        predict([text])
    samples = []
    for text in sample:
        if device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        predict([text])
        if device == "mps":
            torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "warm_single_mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "sample_count": len(samples),
        "device": device,
        "conditions": "20 sequential single-comment requests after 3 warmups, tokenizer and model inference included, no HTTP overhead, MPS synchronized; dynamic length <=256.",
    }


def train_baselines(
    records: list[dict], split: dict, *, seeds: tuple[int, ...] = (42, 7, 2024)
) -> list[dict]:
    parts = partitions(records, split)
    train_text = [r["text"] for r in parts["train"]]
    train_labels = [r["label"] for r in parts["train"]]
    runs = []
    for kind in ["majority", "tfidf"]:
        for seed in seeds:
            model_id = f"{kind}-seed{seed}"
            config = {
                "seed": seed,
                "preprocessing_fit": "training partition only",
                "max_length": None,
            }
            classifier = (
                DummyClassifier(strategy="most_frequent")
                if kind == "majority"
                else LogisticRegression(C=2.0, max_iter=1000, random_state=seed)
            )
            pipeline = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True
                        ),
                    ),
                    ("classifier", classifier),
                ]
            )
            config.update(
                {
                    "classifier": classifier.get_params(),
                    "tfidf": {
                        "ngram_range": [1, 2],
                        "min_df": 2,
                        "max_features": 30000,
                        "sublinear_tf": True,
                    },
                }
            )
            started = time.perf_counter()
            pipeline.fit(train_text, train_labels)
            duration = time.perf_counter() - started
            metadata = base_metadata(model_id, split, "cpu", config)
            metadata["training_duration_seconds"] = duration
            metadata["display_name"] = (
                "Majority class" if kind == "majority" else "TF-IDF + logistic regression"
            )
            folder = settings.artifacts / model_id
            folder.mkdir(parents=True, exist_ok=True)
            # Only representative seed42 artifacts are retained; every seed retains metrics and row predictions.
            if seed == 42:
                joblib.dump(pipeline, folder / "model.joblib")
                loaded = joblib.load(folder / "model.joblib")
                if not np.array_equal(
                    pipeline.predict(train_text[:3]), loaded.predict(train_text[:3])
                ):
                    raise AssertionError("Baseline reload changed predictions.")
                (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
                update_registry(registry_entry(model_id, "sklearn", metadata))
            run = {
                "model_id": model_id,
                "configuration": kind,
                "seed": seed,
                "training_duration_seconds": duration,
                "metadata": metadata,
            }
            for name in ["validation", "test"]:
                predictions = pipeline.predict([r["text"] for r in parts[name]]).tolist()
                run[name] = compute_metrics([r["label"] for r in parts[name]], predictions)
                save_predictions(model_id, name, parts[name], predictions)
            run["latency"] = measure_latency(
                pipeline.predict, [r["text"] for r in parts["test"]], "cpu"
            )
            runs.append(run)
    return runs


def _model_revision() -> str:
    source = json.loads((settings.root / "configs/model_sources.json").read_text())["student"]
    path = settings.local / "distilbert_revision.json"
    if not path.exists():
        path.write_text(json.dumps({"model": BASE_MODEL, "revision": source["revision"]}))
    revision = json.loads(path.read_text())["revision"]
    if revision != source["revision"] or BASE_MODEL != source["model_id"]:
        raise ValueError("Cached base model differs from configs/model_sources.json.")
    return revision


def train_transformer(
    records: list[dict],
    split: dict,
    *,
    seed: int = 42,
    epochs: int = 3,
    weighted: bool = False,
    device: str = "auto",
    configuration: str = "distilbert-original",
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    selected_device = choose_device(device)
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if selected_device == "mps":
        torch.mps.manual_seed(seed)
    model_id = f"{configuration}-seed{seed}"
    revision = _model_revision()
    config = {
        "seed": seed,
        "epochs": epochs,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "max_length": 256,
        "dynamic_padding": True,
        "class_weighted_loss": weighted,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_clip_norm": 1.0,
        "checkpoint_selection": "highest validation macro F1, earliest epoch wins ties",
        "base_model": BASE_MODEL,
        "base_revision": revision,
        "base_license": "apache-2.0",
        "randomness": "Training seeds vary initialization, dropout, shuffle on fixed thread partitions; MPS kernels are not guaranteed bitwise deterministic.",
    }
    metadata = base_metadata(model_id, split, selected_device, config)
    metadata["display_name"] = (
        f"DistilBERT {'class-weighted' if weighted else str(epochs) + ' epochs'}"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        revision=revision,
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        attn_implementation="eager",
    ).to(selected_device)
    parts = partitions(records, split)
    encoded_parts = {}
    truncation_counts = {}
    for name, subset in parts.items():
        texts = [r["text"] for r in subset]
        encoded = tokenizer(texts, truncation=True, max_length=256)
        full_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in texts
        ]
        truncation_counts[name] = sum(length > 256 for length in full_lengths)
        encoded_parts[name] = [
            {
                **{key: value[i] for key, value in encoded.items()},
                "labels": LABEL2ID[record["label"]],
            }
            for i, record in enumerate(subset)
        ]

    def collate(batch):
        return tokenizer.pad(batch, padding=True, return_tensors="pt")

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        encoded_parts["train"], batch_size=16, shuffle=True, collate_fn=collate, generator=generator
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    counts = np.bincount([LABEL2ID[r["label"]] for r in parts["train"]], minlength=3)
    weights = (
        torch.tensor(
            len(parts["train"]) / (len(LABELS) * counts),
            dtype=torch.float32,
            device=selected_device,
        )
        if weighted
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    def predict_encoded(name: str) -> tuple[list[str], list[list[float]]]:
        model.eval()
        scores = []
        with torch.inference_mode():
            for batch in DataLoader(
                encoded_parts[name], batch_size=16, shuffle=False, collate_fn=collate
            ):
                batch.pop("labels")
                inputs = {key: value.to(selected_device) for key, value in batch.items()}
                scores.extend(torch.softmax(model(**inputs).logits, dim=-1).cpu().tolist())
        return [LABELS[int(np.argmax(row))] for row in scores], scores

    best_f1 = -1.0
    best_state = None
    history = []
    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            labels = batch.pop("labels").to(selected_device)
            inputs = {key: value.to(selected_device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        predicted, _ = predict_encoded("validation")
        metrics = compute_metrics([r["label"] for r in parts["validation"]], predicted)
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": metrics}
        )
        print(
            f"{model_id} epoch {epoch + 1}/{epochs} loss={np.mean(losses):.4f} val_macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            metadata["selected_epoch"] = epoch + 1
    duration = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    model.eval()
    metadata.update(
        {
            "training_duration_seconds": duration,
            "truncation_counts": truncation_counts,
            "class_weights": weights.cpu().tolist() if weights is not None else None,
            "epoch_history": history,
        }
    )
    run = {
        "model_id": model_id,
        "configuration": configuration,
        "seed": seed,
        "training_duration_seconds": duration,
        "metadata": metadata,
        "epoch_history": history,
    }
    for name in ["validation", "test"]:
        predicted, scores = predict_encoded(name)
        run[name] = compute_metrics([r["label"] for r in parts[name]], predicted)
        save_predictions(model_id, name, parts[name], predicted, scores)

    def predict_texts(texts):
        with torch.inference_mode():
            inputs = tokenizer(
                texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(selected_device)
            return model(**inputs).logits

    run["latency"] = measure_latency(
        predict_texts, [r["text"] for r in parts["test"]], selected_device
    )
    if seed == 42:
        folder = settings.artifacts / model_id
        folder.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(folder, safe_serialization=True)
        tokenizer.save_pretrained(folder)
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
        expected = (
            predict_texts(["Could you explain how the cache handles updates?"]).detach().cpu()
        )
        reloaded = (
            AutoModelForSequenceClassification.from_pretrained(
                folder, local_files_only=True, attn_implementation="eager"
            )
            .to(selected_device)
            .eval()
        )
        if reloaded.config.label2id != LABEL2ID:
            raise AssertionError("Saved transformer label mapping changed.")
        with torch.inference_mode():
            inputs = tokenizer(
                ["Could you explain how the cache handles updates?"], return_tensors="pt"
            ).to(selected_device)
            actual = reloaded(**inputs).logits.cpu()
        if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-4):
            raise AssertionError("Saved transformer logits changed after reload.")
        run["reload_verified"] = True
        del reloaded
        update_registry(registry_entry(model_id, "transformer", metadata))
    del optimizer, best_state
    model = None
    if selected_device == "mps":
        torch.mps.empty_cache()
    return run
