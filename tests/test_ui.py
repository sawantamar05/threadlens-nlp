"""Application state tests with a stub HTTP boundary and no model downloads."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "ui" / "app.py"
LABELS = ["analysis", "request", "commentary"]


def control(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture
def api_stub(monkeypatch):
    calls = []
    revisions = []
    model = {
        "id": "test-tfidf",
        "name": "Test TF-IDF",
        "kind": "tfidf",
        "version": "test-1",
        "provenance": "synthetic_demo",
        "available": True,
        "metadata": {},
    }
    prediction = {
        "label": "request",
        "scores": {"analysis": 0.1, "request": 0.8, "commentary": 0.1},
        "model_id": model["id"],
        "model_version": model["version"],
        "truncated": False,
        "processing_ms": 1.3,
        "warnings": [],
        "low_confidence": False,
    }
    metrics = {
        "accuracy": 0.5,
        "macro_f1": 0.4,
        "per_label": {
            label: {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 4}
            for label in LABELS
        },
        "confusion_matrix": [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
    }
    report = {
        "dataset": {
            "comment_count": 12,
            "thread_count": 3,
            "label_counts": dict.fromkeys(LABELS, 4),
            "provenance": "weak_label",
        },
        "runs": [
            {
                "model_id": "test-tfidf",
                "configuration": "tfidf",
                "seed": 42,
                "validation": metrics,
                "test": metrics,
                "latency": {"warm_single_mean_ms": 1.3, "device": "cpu"},
            }
        ],
        "summaries": [{"configuration": "tfidf", "macro_f1_mean": 0.4, "macro_f1_std": 0}],
        "improvement": {"change": "example test fixture", "reason": "UI test only"},
        "quality_status": "Independent semantic accuracy unmeasured",
    }

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        path = httpx.URL(url).path
        if path == "/models":
            data = {
                "active_model": model["id"],
                "models": [model, {**model, "id": "test-majority", "name": "Test majority"}],
            }
        elif path == "/predict":
            data = prediction
        elif path == "/experiments":
            data = report
        elif path == "/review/next":
            data = {
                "item": {
                    "comment_id": "demo-1",
                    "thread_id": "demo",
                    "text": "Could you explain the cache?",
                    "revision_count": len(revisions),
                    "latest_annotation": revisions[-1] if revisions else None,
                }
            }
        elif path == "/annotations":
            data = {
                **kwargs["json"],
                "revision": len(revisions) + 1,
                "timestamp": "test-time",
                "review_status": "reviewed",
            }
            revisions.append(data)
        elif path == "/annotations/demo-1":
            data = {
                "comment_id": "demo-1",
                "draft": {"label": "request", "provenance": "synthetic_demo"}
                if revisions
                else None,
                "revisions": revisions,
            }
        else:
            return httpx.Response(404, json={"detail": "Missing test endpoint"})
        return httpx.Response(200, json=data)

    monkeypatch.setattr(httpx, "request", request)
    return calls, revisions


def test_empty_input_and_prediction_then_model_change(api_stub):
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    control(app.button, "Classify comment").click().run()
    assert any("non-empty comment" in error.value for error in app.error)
    assert not any(url.endswith("/predict") for _, url, _ in api_stub[0])
    control(app.text_area, "Complete comment").set_value("Could you explain the cache?")
    control(app.button, "Classify comment").click().run()
    assert not app.exception
    assert any(metric.value == "Request" for metric in app.metric)
    control(app.selectbox, "Select model").select("test-majority").run()
    assert not app.exception
    assert not app.metric


def test_experiments_show_provenance_and_actual_run_tables(api_stub):
    app = AppTest.from_file(str(APP)).run()
    control(app.radio, "Workspace").set_value("Experiments").run()
    assert not app.exception
    assert any("not independently validated" in warning.value for warning in app.warning)
    assert len(app.dataframe) >= 3
    assert any(metric.label == "Comments" and metric.value == "12" for metric in app.metric)
    assert app.get("plotly_chart")


def test_blind_review_and_revision_reload_remain_ai_assisted(api_stub):
    app = AppTest.from_file(str(APP)).run()
    control(app.radio, "Workspace").set_value("Annotation review").run()
    assert not app.exception
    assert not app.get("json")
    control(app.radio, "Assign a label").set_value("request")
    control(app.button, "Save annotation").click().run()
    assert not app.exception
    assert api_stub[1][0]["provenance"] == "ai_assisted_reviewed"
    assert api_stub[1][0]["blind"] is True
    assert any("Saved revision 1" in success.value for success in app.success)
    control(app.button, "Reload saved revisions").click().run()
    assert not app.exception
    assert len(app.dataframe) == 1


def test_missing_api_is_actionable_and_disables_inference(monkeypatch):
    def disconnected(*args, **kwargs):
        raise httpx.ConnectError("API offline")

    monkeypatch.setattr(httpx, "request", disconnected)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert control(app.button, "Classify comment").disabled
    assert any("local API is unavailable" in error.value for error in app.error)
