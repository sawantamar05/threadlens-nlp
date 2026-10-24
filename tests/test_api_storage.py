"""Network-free contract checks using a genuinely fitted tiny sklearn model."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from threadlens.api.app import create_app
from threadlens.config import LABELS, Settings
from threadlens.data.csv_io import parse_csv, safe_csv
from threadlens.models.inference import ModelManager, ModelUnavailable
from threadlens.storage.annotations import AnnotationStore


@pytest.fixture
def config(tmp_path: Path) -> Settings:
    config = Settings(root=tmp_path)
    directory = config.artifacts / "tiny"
    directory.mkdir(parents=True)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(C=10, random_state=42)),
        ]
    )
    model.fit(
        [
            "measurement mechanism observed benchmark",
            "comparison measured evidence mechanism",
            "please explain question advice",
            "question could anyone explain please",
            "wonderful joke lovely opinion",
            "lovely reaction wonderful joke",
        ],
        ["analysis", "analysis", "request", "request", "commentary", "commentary"],
    )
    joblib.dump(model, directory / "model.joblib")
    (config.artifacts / "model_registry.json").write_text(
        json.dumps(
            {
                "active_model": "tiny",
                "models": [
                    {
                        "id": "tiny",
                        "name": "Tiny fitted test baseline",
                        "kind": "sklearn",
                        "path": "tiny/model.joblib",
                        "version": "test-v1",
                        "provenance": "synthetic_demo",
                        "metadata": {},
                    }
                ],
            }
        )
    )
    return config


@pytest.fixture
def client(config):
    with TestClient(create_app(config)) as client:
        yield client


def test_health_openapi_and_missing_model(tmp_path):
    with TestClient(create_app(Settings(root=tmp_path))) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/models").json() == {"active_model": None, "models": []}
        response = client.get("/ready")
        assert response.status_code == 503
        assert "threadlens demo" in response.json()["detail"]
        assert client.post("/predict", json={"text": "hello world"}).status_code == 503
        assert "/annotations" in client.get("/openapi.json").json()["paths"]


def test_prediction_mapping_reload_and_cached_instance(client, config):
    response = client.post("/predict", json={"text": "please explain question advice"})
    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "request"
    assert set(result["scores"]) == set(LABELS)
    assert sum(result["scores"].values()) == pytest.approx(1)
    assert result["model_version"] == "test-v1"
    assert result["processing_ms"] >= 0
    assert result["truncated"] is False
    assert any("Synthetic" in warning for warning in result["warnings"])
    manager = client.app.state.manager
    instance = manager._loaded["tiny"]["model"]
    manager.predict(["joke lovely"])
    assert manager._loaded["tiny"]["model"] is instance
    reloaded = ModelManager(config).predict(["please explain question advice"])[0]
    assert reloaded["scores"] == result["scores"]


@pytest.mark.parametrize("text", ["", " \n\t", "x" * 10001])
def test_reject_invalid_single_text(client, text):
    assert client.post("/predict", json={"text": text}).status_code == 422


def test_batch_bounds_and_order(client):
    texts = ["lovely reaction joke", "please question explain", "measurement observed mechanism"]
    results = client.post("/predict/batch", json={"texts": texts}).json()["results"]
    assert [result["label"] for result in results] == ["commentary", "request", "analysis"]
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["abc"] * 101}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["abc", "  "]}).status_code == 422
    assert client.post("/predict", json={"text": "abc", "model_id": "missing"}).status_code == 503


def test_request_body_bounded(client):
    response = client.post("/predict", content="x" * (2 * 1024 * 1024 + 1))
    assert response.status_code == 413


def test_registry_prevents_artifact_path_escape(config):
    registry = config.artifacts / "model_registry.json"
    data = json.loads(registry.read_text())
    data["models"][0]["path"] = "../../secrets.joblib"
    registry.write_text(json.dumps(data))
    with pytest.raises(ModelUnavailable, match="inside"):
        ModelManager(config).models()


def test_csv_validation_preserves_order_and_literal_na():
    frame = parse_csv(b'id,text\n3,"hello, world"\n1,NA\n2,third row\n')
    assert frame.id.tolist() == ["3", "1", "2"]
    assert frame.text.tolist() == ["hello, world", "NA", "third row"]
    cases = [
        b"",
        b"id\n1\n",
        b"text\n\n",
        b"text,text\na,b\n",
        b'text\n"unterminated',
        b"text\nhello,extra\n",
    ]
    for value in cases:
        with pytest.raises(ValueError):
            parse_csv(value)
    with pytest.raises(ValueError, match="UTF-8"):
        parse_csv(b"text\n\xff")
    with pytest.raises(ValueError, match="1000"):
        parse_csv(b"text\n" + b"row\n" * 1001)
    with pytest.raises(ValueError, match="10000"):
        parse_csv(b"text\n" + b"x" * 10001)
    with pytest.raises(ValueError, match="MB"):
        parse_csv(b"text\n" + b"x" * (2 * 1024 * 1024))


def test_formula_export_sanitizes_headers_values_and_keeps_rows():
    frame = pd.DataFrame({"=header": ["=1+1", " \t@SUM(A1)", "+cmd", "-2+3", "\tplain", "normal"]})
    result = safe_csv(frame)
    assert result.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(result.decode("utf-8-sig"))))
    assert rows[0] == ["'=header"]
    assert rows[1:] == [["'=1+1"], ["' \t@SUM(A1)"], ["'+cmd"], ["'-2+3"], ["'\tplain"], ["normal"]]
    assert frame.iloc[0, 0] == "=1+1"


def test_original_draft_preserved_and_blind_revision_provenance(tmp_path):
    store = AnnotationStore(tmp_path / "state.db")
    record = {
        "comment_id": 1,
        "thread_id": 10,
        "text": "Can anyone explain?",
        "label": "analysis",
        "scores": {"analysis": 0.8},
        "provenance": "weak_label",
    }
    assert store.import_records([record]) == 1
    assert store.next_item(blind=True)["draft_label"] is None
    assert store.history("1", blind=True)["draft"] is None
    initial = store.save("1", "request", "human_independent", blind=True)
    assert initial["independent_initial"] == 1
    assert initial["provenance"] == "human_independent"
    assert store.history("1")["draft"]["label"] == "analysis"
    store.save("1", "commentary", "ai_assisted_reviewed", blind=False, notes="revision")
    store.import_records([{**record, "label": "commentary"}])
    history = AnnotationStore(tmp_path / "state.db").history("1")
    assert history["draft"]["label"] == "analysis"
    assert [revision["label"] for revision in history["revisions"]] == ["request", "commentary"]
    assert history["revisions"][0]["provenance"] == "human_independent"
    with pytest.raises(ValueError, match="initial blind"):
        store.save("1", "analysis", "human_independent", blind=True)


def test_suggestion_exposure_cannot_become_independent(tmp_path):
    store = AnnotationStore(tmp_path / "review.db")
    store.import_records(
        [{"comment_id": "2", "text": "Hello", "label": "commentary", "provenance": "weak_label"}]
    )
    assert store.next_item(blind=False)["draft_label"] == "commentary"
    assert store.next_item(blind=True)["draft_label"] is None
    with pytest.raises(ValueError, match="suggestion"):
        store.save("2", "commentary", "human_independent", blind=True)
    assert (
        store.save("2", "commentary", "ai_assisted_reviewed")["provenance"]
        == "ai_assisted_reviewed"
    )


def test_synthetic_annotations_stay_synthetic(client):
    client.app.state.store.import_records(
        [
            {
                "comment_id": "demo-1",
                "text": "Synthetic original",
                "label": "commentary",
                "provenance": "synthetic_demo",
            }
        ]
    )
    response = client.post(
        "/annotations",
        json={
            "comment_id": "demo-1",
            "label": "analysis",
            "provenance": "human_independent",
            "blind": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["provenance"] == "synthetic_demo"
    history = client.get("/annotations/demo-1").json()
    assert len(history["revisions"]) == 1
    assert client.get("/annotations/unknown").status_code == 404
    assert client.get("/review/status").json()["human_independent"] == 0


def test_thread_validation_and_network_failure(client, monkeypatch):
    from threadlens.data import hn

    for identifier in [
        "https://evil.test/item?id=1",
        "https://news.ycombinator.com.evil.test/item?id=1",
        "-1",
        "0",
        "https://news.ycombinator.com/item?id=1&id=2",
    ]:
        assert client.post("/analyze-thread", json={"story": identifier}).status_code == 422

    def fail(*args):
        raise RuntimeError("network disconnected")

    monkeypatch.setattr(hn, "fetch_story_comments", fail)
    response = client.post("/analyze-thread", json={"story": "123"})
    assert response.status_code == 502
    assert "unavailable" in response.json()["detail"]


def test_thread_counts_reviews_and_empty_states(client, monkeypatch):
    from threadlens.data import hn

    comments = [
        {
            "comment_id": 1,
            "thread_id": 123,
            "text": "please explain question",
            "permalink": "https://news.ycombinator.com/item?id=1",
        },
        {
            "comment_id": 2,
            "thread_id": 123,
            "text": "joke lovely reaction",
            "permalink": "https://news.ycombinator.com/item?id=2",
        },
    ]
    monkeypatch.setattr(
        hn,
        "fetch_story_comments",
        lambda *args: {"story_id": 123, "title": "Test", "comments": comments, "warnings": []},
    )
    client.app.state.store.import_records(
        [{**comments[0], "label": "analysis", "provenance": "weak_label"}]
    )
    client.app.state.store.save("1", "request", "ai_assisted_reviewed")
    response = client.post("/analyze-thread", json={"story": "123", "limit": 2})
    assert response.status_code == 200
    result = response.json()
    assert result["counts"] == {"analysis": 0, "request": 1, "commentary": 1}
    assert sum(result["proportions"].values()) == 1
    assert result["comments"][0]["reviewed_label"] == "request"
    assert result["comments"][0]["reviewed_provenance"] == "ai_assisted_reviewed"
    comments.clear()
    empty = client.post("/analyze-thread", json={"story": 123}).json()
    assert empty["comments"] == []
    assert any("No usable" in value for value in empty["warnings"])


def test_experiment_report_missing_and_valid(client, config):
    assert client.get("/experiments").status_code == 404
    config.reports.mkdir()
    (config.reports / "experiments.json").write_text(
        json.dumps({"quality_status": "provisional weak-label agreement"})
    )
    assert "provisional" in client.get("/experiments").json()["quality_status"]


def test_chunked_body_is_bounded_without_content_length(client):
    def body():
        yield b'{"text":"'
        for _ in range(33):
            yield b"x" * 65536
        yield b'"}'

    response = client.post("/predict", content=body(), headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert "input" not in response.text


def test_receive_timeout_and_actual_byte_count_before_dispatch():
    import asyncio

    from threadlens.api.middleware import RequestBodyLimitMiddleware

    async def check(messages, headers=(), slow=False):
        dispatched = []
        sent = []

        async def app(scope, receive, send):
            dispatched.append(await receive())

        async def receive():
            if slow:
                await asyncio.sleep(0.03)
            return messages.pop(0)

        async def send(message):
            sent.append(message)

        middleware = RequestBodyLimitMiddleware(app, max_bytes=5, timeout_seconds=0.01)
        await middleware({"type": "http", "headers": list(headers)}, receive, send)
        return dispatched, sent

    # A lying short Content-Length cannot bypass counting the actual streamed body.
    dispatched, sent = asyncio.run(
        check(
            [
                {"type": "http.request", "body": b"123", "more_body": True},
                {"type": "http.request", "body": b"456", "more_body": False},
            ],
            [(b"content-length", b"1")],
        )
    )
    assert dispatched == []
    assert sent[0]["status"] == 413
    dispatched, sent = asyncio.run(
        check(
            [
                {"type": "http.request", "body": b"123", "more_body": True},
                {"type": "http.request", "body": b"45", "more_body": False},
            ]
        )
    )
    assert dispatched[0]["body"] == b"12345"
    assert sent == []
    dispatched, sent = asyncio.run(
        check(
            [
                {"type": "http.request", "body": b"", "more_body": False},
            ],
            slow=True,
        )
    )
    assert dispatched == []
    assert sent[0]["status"] == 408
    _, sent = asyncio.run(check([], [(b"content-length", b"-1")]))
    assert sent[0]["status"] == 400


@pytest.mark.parametrize("text", ["\u00a0\u2003\t\r\n", "\u200b\ufeff", "hello\x00world"])
def test_unicode_blank_and_null_text_rejected(client, text):
    assert client.post("/predict", json={"text": text}).status_code == 422
    assert client.post("/predict/batch", json={"texts": [text]}).status_code == 422
    with pytest.raises(ValueError):
        parse_csv(f'text\n"{text}"\n'.encode())


def test_validation_responses_do_not_echo_rejected_source_text(client):
    secret = "PRIVATE-LOCAL-COMMENT-" + "x" * 10000
    result = client.post("/predict", json={"text": secret})
    assert result.status_code == 422
    assert "PRIVATE-LOCAL-COMMENT" not in result.text
    assert "input" not in result.json()["detail"][0]


def test_loaded_once_under_concurrent_prediction_requests(config, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    from threadlens.models import inference

    original = inference.joblib.load
    loads = []

    def load(path):
        loads.append(path)
        time.sleep(0.01)
        return original(path)

    monkeypatch.setattr(inference.joblib, "load", load)
    manager = ModelManager(config)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(lambda _: manager.predict(["please question advice"])[0], range(16))
        )
    assert len(loads) == 1
    assert all(result["label"] == "request" for result in results)


def test_changed_artifact_is_reloaded_instead_of_reported_with_stale_weights(config):
    from sklearn.dummy import DummyClassifier

    manager = ModelManager(config)
    assert manager.predict(["please explain question"])[0]["label"] == "request"
    path = config.artifacts / "tiny/model.joblib"
    replacement = DummyClassifier(strategy="most_frequent").fit(
        [[0], [0]], ["commentary", "commentary"]
    )
    joblib.dump(replacement, path)
    registry_path = config.artifacts / "model_registry.json"
    registry = json.loads(registry_path.read_text())
    registry["models"][0]["version"] = "test-v2"
    registry_path.write_text(json.dumps(registry))
    result = manager.predict(["please explain question"])[0]
    assert result["label"] == "commentary"
    assert result["model_version"] == "test-v2"
    assert result["scores"] == {"analysis": 0, "request": 0, "commentary": 1}


def test_invalid_registry_and_corrupted_model_are_actionable_503(client, config):
    path = config.artifacts / "model_registry.json"
    original = path.read_text()
    path.write_text('{"models":[{"id":"broken"}]}')
    assert client.get("/models").status_code == 503
    path.write_text(original)
    (config.artifacts / "tiny/model.joblib").write_bytes(b"not a pickle")
    assert client.get("/ready").status_code == 503


def test_majority_subset_and_numeric_label_mappings(config):
    from sklearn.dummy import DummyClassifier

    path = config.artifacts / "tiny/model.joblib"
    for labels, expected in [
        (["request", "request", "analysis"], "request"),
        ([2, 2, 1], "commentary"),
    ]:
        joblib.dump(DummyClassifier(strategy="most_frequent").fit([[0]] * 3, labels), path)
        result = ModelManager(config).predict(["unseen entirely"])[0]
        assert result["label"] == expected
        assert result["scores"][expected] == 1
        assert set(result["scores"]) == set(LABELS)
    joblib.dump(DummyClassifier(strategy="most_frequent").fit([[0]] * 2, [-1, -1]), path)
    with pytest.raises(ModelUnavailable, match="mapping"):
        ModelManager(config).predict(["hello"])


def test_inference_elsewhere_prevents_later_independent_claim(client):
    text = "please explain question advice"
    client.app.state.store.import_records(
        [
            {
                "comment_id": "external-exposure",
                "text": text,
                "label": "request",
                "provenance": "weak_label",
            }
        ]
    )
    assert client.get("/review/next", params={"blind": True}).json()["item"]["draft_label"] is None
    assert client.post("/predict", json={"text": text}).status_code == 200
    rejected = client.post(
        "/annotations",
        json={
            "comment_id": "external-exposure",
            "label": "request",
            "provenance": "human_independent",
            "blind": True,
        },
    )
    assert rejected.status_code == 422
    assert client.app.state.store.history("external-exposure")["revisions"] == []
    accepted = client.post(
        "/annotations",
        json={
            "comment_id": "external-exposure",
            "label": "request",
            "provenance": "ai_assisted_reviewed",
            "blind": True,
        },
    ).json()
    assert accepted["independent_initial"] == 0


def test_sqlite_connections_close_and_failed_revision_rolls_back(tmp_path):
    import sqlite3

    store = AnnotationStore(tmp_path / "state.db")
    with store.connect() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    store.import_records(
        [{"comment_id": "a", "text": "test text", "label": "request", "provenance": "weak_label"}]
    )
    with pytest.raises(ValueError):
        store.save("a", "request", "human_independent", blind=False)
    assert store.history("a")["revisions"] == []
    assert store.save("a", "request", "human_independent", blind=True)["revision"] == 1


@pytest.mark.parametrize(
    "story",
    [
        None,
        {"type": "comment"},
        {"type": "story", "deleted": True},
        {"type": "story", "dead": True},
    ],
)
def test_actual_hn_client_rejects_missing_deleted_dead_nonstory(client, monkeypatch, story):
    from threadlens.data import hn
    from threadlens.data.hn import HNClient

    monkeypatch.setattr(hn, "settings", client.app.state.settings)
    monkeypatch.setattr(HNClient, "item", lambda *args: story)
    assert client.post("/analyze-thread", json={"story": "123"}).status_code == 422


def test_thread_timeout_is_informative_and_noninteger_ids_rejected(client, monkeypatch):
    import httpx

    from threadlens.data import hn

    def timeout(*args):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(hn, "fetch_story_comments", timeout)
    response = client.post("/analyze-thread", json={"story": "123"})
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]
    for value in [True, 1.5, [123], {"id": 123}]:
        assert client.post("/analyze-thread", json={"story": value}).status_code == 422


def test_neural_truncation_uses_full_input_for_detection_without_mutating_source(
    config, monkeypatch
):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Encoded(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            lengths = [len(text.split()) + 2 for text in texts]
            if kwargs.get("return_length"):
                return {"length": lengths}
            assert kwargs["max_length"] == 256
            return Encoded(lengths=[min(length, 256) for length in lengths])

    class Model:
        def __call__(self, lengths):
            return SimpleNamespace(logits=np.array([[0, 1, 0] for _ in lengths]))

    def softmax(logits, dim):
        values = np.exp(logits)
        return Tensor(values / values.sum(axis=1, keepdims=True))

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext, softmax=softmax)
    )
    texts = ["word " * 254, "word " * 255, "short input"]
    original = list(texts)
    manager = ModelManager(config)
    entry = {"id": "mock-neural", "kind": "transformer", "metadata": {"max_length": 256}}
    scores, truncated = manager._neural(
        texts, entry, {"tokenizer": Tokenizer(), "model": Model(), "device": "cpu"}
    )
    assert texts == original
    assert truncated == [False, True, False]
    assert scores.shape == (3, 3)
    assert manager._metadata(entry)["max_length"] == 256
    assert "full source" in manager._metadata(entry)["input_policy"]


@pytest.mark.parametrize(
    "scores", [[[float("nan"), 0, 0]], [[-0.1, 0.6, 0.5]], [[0.2, 0.2, 0.2]], [[1, 0]]]
)
def test_invalid_model_scores_never_become_successful_predictions(config, monkeypatch, scores):
    from types import SimpleNamespace

    manager = ModelManager(config)
    model = SimpleNamespace(classes_=list(LABELS), predict_proba=lambda texts: scores)
    monkeypatch.setattr(
        manager, "_load", lambda entry: {"model": model, "device": "cpu", "warnings": []}
    )
    with pytest.raises(ModelUnavailable, match="scores"):
        manager.predict(["a complete comment"])


def test_transformer_load_enforces_both_label_mappings_and_uses_only_local_files(
    config, monkeypatch
):
    import sys
    from types import SimpleNamespace

    directory = config.artifacts / "local-transformer"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    (directory / "tokenizer.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"fixture-never-deserialized")
    calls = []

    class Model:
        config = SimpleNamespace(
            id2label=dict(enumerate(LABELS)),
            label2id={label: index for index, label in enumerate(LABELS)},
        )

        def to(self, device):
            return self

        def eval(self):
            return self

    model = Model()

    def load_model(path, **kwargs):
        calls.append((path, kwargs))
        return model

    def load_tokenizer(path, **kwargs):
        calls.append((path, kwargs))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=load_model),
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        ),
    )
    monkeypatch.setenv("THREADLENS_DEVICE", "auto")
    registry = {
        "active_model": "local-transformer",
        "models": [
            {
                "id": "local-transformer",
                "kind": "transformer",
                "path": "local-transformer",
                "version": "fixture",
                "metadata": {"max_length": 256},
            }
        ],
    }
    (config.artifacts / "model_registry.json").write_text(json.dumps(registry))
    manager = ModelManager(config)
    ready = manager.ensure_ready()
    assert ready["device"] == "cpu"
    assert "MPS is unavailable" in ready["warnings"][0]
    assert all(kwargs == {"local_files_only": True} for _, kwargs in calls)
    assert manager.models()["models"][0]["metadata"]["max_length"] == 256
    assert manager.models()["models"][0]["metadata"]["inference_device"] == "cpu"
    model.config.label2id = {"analysis": 1, "request": 0, "commentary": 2}
    with pytest.raises(ModelUnavailable, match="mapping"):
        ModelManager(config).ensure_ready()


def test_service_bounds_apply_outside_http(config):
    from threadlens.validation import InputInvalid

    manager = ModelManager(config)
    for texts in [[], ["\u200b"], ["x"] * 101, ["x" * 10001], ["hello\x00world"]]:
        with pytest.raises(InputInvalid):
            manager.predict(texts)
    # Custom settings can tighten the service bound without bypass through the CLI.
    smaller = Settings(root=config.root, max_chars=5)
    with TestClient(create_app(smaller)) as client:
        assert client.post("/predict", json={"text": "123456"}).status_code == 422
