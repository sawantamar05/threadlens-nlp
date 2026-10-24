"""Lightweight deterministic checks; no model download or external network."""

from __future__ import annotations

import json

import joblib
import pytest

from threadlens.config import ID2LABEL, LABEL2ID, LABELS, Settings
from threadlens.data.hn import HNClient, clean_html, parse_story_id, text_hash
from threadlens.data.splits import create_splits
from threadlens.evaluation.metrics import compute_metrics


def records():
    return [
        {
            "comment_id": thread * 3 + i,
            "thread_id": thread,
            "text": f"{label} signal example group {thread}",
            "text_hash": f"hash-{thread}-{i}",
            "label": label,
            "provenance": "synthetic_demo",
        }
        for thread in range(20)
        for i, label in enumerate(LABELS)
    ]


def test_html_paragraphs_and_entities():
    assert (
        clean_html("first<p>second &amp; third<br>line</p><script>danger</script>")
        == "first\n\nsecond & third\nline"
    )


def test_duplicate_hash_ignores_whitespace_but_preserves_content():
    assert text_hash(" one\n two ") == text_hash("one two")
    assert text_hash("one two") != text_hash("one three")


@pytest.mark.parametrize(
    "value",
    [
        "https://evil.test/item?id=1",
        "http://news.ycombinator.com/item?id=1",
        "https://news.ycombinator.com/item?id=1&id=2",
        "0",
        "-1",
        "1.2",
        "https://news.ycombinator.com@evil.test/item?id=1",
    ],
)
def test_story_identifiers_reject_arbitrary_fetch(value):
    with pytest.raises(ValueError):
        parse_story_id(value)


def test_valid_story_identifiers():
    assert parse_story_id("https://news.ycombinator.com/item?id=123") == 123
    assert parse_story_id(" 123 ") == 123


def test_cache_resumes_without_network(tmp_path, monkeypatch):
    path = tmp_path / "cache.sqlite3"
    client = HNClient(path)
    monkeypatch.setattr(client, "_get", lambda url: {"id": 100, "type": "story"})
    assert client.item(100)["id"] == 100
    offline = HNClient(path, offline=True)
    assert offline.item(100)["type"] == "story"
    with pytest.raises(RuntimeError, match="not cached"):
        offline.item(200)


def test_deleted_missing_and_empty_comments_filtered(tmp_path, monkeypatch):
    values = {
        100: {"type": "story", "kids": [101, 102, 103, 104, 105]},
        101: {"id": 101, "type": "comment", "text": "<p>Hello world</p>"},
        102: {"id": 102, "type": "comment", "deleted": True, "text": "hidden"},
        103: None,
        104: {"id": 104, "type": "comment", "dead": True, "text": "hidden"},
        105: {"id": 105, "type": "comment", "text": " "},
    }
    client = HNClient(tmp_path / "cache.sqlite3")
    monkeypatch.setattr(client, "item", lambda identifier: values[identifier])
    assert [r["comment_id"] for r in client.story_comments(100, 20)] == [101]
    with pytest.raises(ValueError):
        client.story_comments(101, 20)


def test_mapping_is_bijective_and_fixed():
    assert LABEL2ID == {"analysis": 0, "request": 1, "commentary": 2}
    assert all(ID2LABEL[LABEL2ID[label]] == label for label in LABELS)


def test_group_isolation_deterministic_and_all_classes_present():
    result = create_splits(records())
    assert result == create_splits(records())
    parts = result["partitions"]
    groups = [set(value["thread_ids"]) for value in parts.values()]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert sum(part["count"] for part in parts.values()) == len(records())
    assert all(set(part["label_counts"]) == set(LABELS) for part in parts.values())


def test_metrics_include_zero_prediction_classes_and_support():
    metrics = compute_metrics(list(LABELS), ["analysis"] * 3)
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["macro_f1"] == pytest.approx(1 / 6)
    assert metrics["per_label"]["request"]["recall"] == 0
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert metrics["prediction_counts"] == {"analysis": 3, "request": 0, "commentary": 0}
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_baseline_reloads_and_vectorizer_never_fits_test_text(tmp_path, monkeypatch):
    from threadlens.models import training

    source = records()
    split = create_splits(source)
    for index in split["partitions"]["test"]["indices"]:
        source[index]["text"] += " heldoutunicorn"
    monkeypatch.setattr(training, "settings", Settings(root=tmp_path))
    monkeypatch.setattr(training, "code_revision", lambda: "test-revision")
    runs = training.train_baselines(source, split, seeds=(42,))
    loaded = joblib.load(tmp_path / "artifacts/tfidf-seed42/model.joblib")
    assert "heldoutunicorn" not in loaded.named_steps["tfidf"].vocabulary_
    assert set(loaded.classes_) == set(LABELS)
    assert len(loaded.predict(["request signal example"])) == 1
    assert {run["configuration"] for run in runs} == {"majority", "tfidf"}
    registry = json.loads((tmp_path / "artifacts/model_registry.json").read_text())
    assert len(registry["models"]) == 2


def test_public_report_does_not_count_as_a_local_model_artifact(tmp_path, monkeypatch):
    from threadlens.evaluation import runner

    monkeypatch.setattr(runner, "settings", Settings(root=tmp_path))
    configuration = "distilbert-original"
    assert not runner.configuration_artifact_available(configuration, "frozen-data")
    folder = tmp_path / "artifacts/distilbert-original-seed42"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps({"dataset_hash": "frozen-data", "model_id": "distilbert-original-seed42"})
    )
    (folder / "model.safetensors").write_bytes(b"unit test placeholder; no actual model load")
    registry = tmp_path / "artifacts/model_registry.json"
    registry.write_text(json.dumps({"models": [{"id": "distilbert-original-seed42"}]}))
    assert runner.configuration_artifact_available(configuration, "frozen-data")
    assert not runner.configuration_artifact_available(configuration, "other-data")
    registry.write_text(json.dumps({"models": []}))
    assert not runner.configuration_artifact_available(configuration, "frozen-data")


def test_improvement_is_bound_to_frozen_dataset(tmp_path, monkeypatch):
    from threadlens.evaluation import runner

    monkeypatch.setattr(runner, "settings", Settings(root=tmp_path))
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_dir.joinpath("improvement.json").write_text(json.dumps({"dataset_hash": "other-data"}))
    runs = [{"configuration": "distilbert-original", "metadata": {"dataset_hash": "frozen-data"}}]
    with pytest.raises(ValueError, match="dataset hash"):
        runner.declare_improvement(runs)
