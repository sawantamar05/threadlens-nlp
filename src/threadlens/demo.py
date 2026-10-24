"""Small original synthetic fixtures, isolated from real data and experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from threadlens.config import ROOT


def prepare_demo(root: Path) -> dict:
    registry_file = root / "artifacts/model_registry.json"
    if registry_file.exists():
        previous = json.loads(registry_file.read_text())
        if any(entry.get("provenance") != "synthetic_demo" for entry in previous.get("models", [])):
            raise ValueError(
                "Refusing to overwrite a non-demo model registry. Choose an isolated demo root."
            )
    source = ROOT / "data/demo/comments.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )
    model.fit([row["text"] for row in rows], [row["label"] for row in rows])
    directory = root / "artifacts/demo-tfidf"
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    metadata = {
        "id": "demo-tfidf",
        "kind": "tfidf",
        "path": "artifacts/demo-tfidf",
        "name": "Synthetic demo · TF-IDF",
        "version": "synthetic-demo-v1",
        "provenance": "synthetic_demo",
        "metadata": {
            "training_rows": len(rows),
            "notice": "Synthetic functional demonstration only. No real benchmark evaluation.",
            "seed": 42,
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    registry = {"active_model": "demo-tfidf", "models": [metadata]}
    (root / "artifacts/model_registry.json").write_text(json.dumps(registry, indent=2))
    (root / ".local").mkdir(exist_ok=True)
    drafts = [
        {
            "comment_id": f"demo-{i}",
            "thread_id": f"demo-thread-{i // 3}",
            "text": row["text"],
            "label": row["label"],
            "provenance": "synthetic_demo",
            "scores": {},
            "review_status": "unreviewed",
            "permalink": None,
        }
        for i, row in enumerate(rows)
    ]
    (root / ".local/labeled.jsonl").write_text("".join(json.dumps(row) + "\n" for row in drafts))
    return registry
