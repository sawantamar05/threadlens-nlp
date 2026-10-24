"""Actual-artifact/API verification; requires running loopback services, no mocks."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/local_integration.json"))
    args = parser.parse_args()
    checks = {}
    with httpx.Client(base_url=args.api, timeout=180) as client:
        assert client.get("/health").status_code == 200
        checks["health"] = True
        models = client.get("/models").json()
        model = args.model or models["active_model"]
        ready = client.get("/ready", params={"model_id": model})
        ready.raise_for_status()
        checks["actual_model_ready"] = ready.json()
        text = "Our benchmark measured 12 milliseconds with caching and 80 without it because repeated requests reused the parsed result."
        single = client.post("/predict", json={"text": text, "model_id": model})
        single.raise_for_status()
        prediction = single.json()
        assert prediction["model_id"] == model
        assert set(prediction["scores"]) == {"analysis", "request", "commentary"}
        assert abs(sum(prediction["scores"].values()) - 1) < 1e-5
        assert not prediction["truncated"]
        checks["single_prediction_on_original_synthetic_input"] = prediction
        for invalid in ("", "   ", "\u200b", "a" * 10001):
            assert client.post("/predict", json={"text": invalid}).status_code == 422
        assert client.post("/predict/batch", json={"texts": ["test"] * 101}).status_code == 422
        checks["invalid_and_bounded_inputs"] = True
        assert (
            client.post("/predict", json={"text": text, "model_id": "missing-artifact"}).status_code
            == 503
        )
        checks["missing_model_actionable_503"] = True
        long = "The cache stores a parsed response. " * 160
        truncated = client.post("/predict", json={"text": long, "model_id": model})
        truncated.raise_for_status()
        kind = next(entry["kind"] for entry in models["models"] if entry["id"] == model)
        if kind == "transformer":
            assert truncated.json()["truncated"]
        checks["long_comment_truncation_flag"] = truncated.json()["truncated"]
        texts = [text, "Could you explain which setting controls retries?", "I like this idea."]
        response = client.post("/predict/batch", json={"texts": texts, "model_id": model})
        response.raise_for_status()
        results = response.json()["results"]
        assert len(results) == len(texts)
        for source, result in zip(texts, results, strict=True):
            expected = client.post("/predict", json={"text": source, "model_id": model}).json()
            assert result["label"] == expected["label"]
            assert all(
                abs(result["scores"][label] - expected["scores"][label]) < 0.001
                for label in expected["scores"]
            )
        checks["actual_model_batch_order_and_score_consistency"] = True
        for invalid_story in ("https://example.com/item?id=1", "0", "-5"):
            assert (
                client.post(
                    "/analyze-thread", json={"story": invalid_story, "limit": 3, "model_id": model}
                ).status_code
                == 422
            )
        checks["arbitrary_urls_rejected"] = True
        manifest = json.loads(Path("reports/collection_manifest.json").read_text())
        story = manifest["records"][0]["thread_id"]
        thread = client.post(
            "/analyze-thread", json={"story": story, "limit": 5, "model_id": model}
        )
        thread.raise_for_status()
        result = thread.json()
        assert 0 < len(result["comments"]) <= 5
        assert sum(result["counts"].values()) == len(result["comments"])
        assert abs(sum(result["proportions"].values()) - 1) < 1e-8
        assert all(row["model_id"] == model for row in result["comments"])
        checks["actual_official_hn_thread"] = {
            "story_id": story,
            "comments": len(result["comments"]),
            "counts": result["counts"],
            "source_text_published": False,
        }
        queue = client.get("/review/next", params={"blind": "true"}).json()["item"]
        if queue:
            assert queue["draft_label"] is None or queue["revision_count"] > 0
            save = client.post(
                "/annotations",
                json={
                    "comment_id": queue["comment_id"],
                    "label": "commentary",
                    "provenance": "ai_assisted_reviewed",
                    "blind": True,
                    "notes": "Automated persistence verification only; not independent semantic validation.",
                },
            )
            save.raise_for_status()
            assert save.json()["provenance"] == "ai_assisted_reviewed"
            assert save.json()["independent_initial"] == 0
            history = client.get(f"/annotations/{queue['comment_id']}").json()
            assert history["revisions"][-1]["revision"] == save.json()["revision"]
            checks["annotation_persistence_and_ai_provenance"] = {
                "saved": True,
                "independent_human": False,
            }
        report = client.get("/experiments")
        report.raise_for_status()
        assert report.json()["benchmark_provenance"] == "weak_label"
        checks["experiment_report_provenance"] = "weak_label"
        schema = client.get("/openapi.json")
        schema.raise_for_status()
        checks["openapi"] = True
    args.output.write_text(
        json.dumps(
            {
                "verified_at": datetime.now(UTC).isoformat(),
                "api": args.api,
                "model_id": model,
                "checks": checks,
                "all_passed": True,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "all_passed": True,
                "model": model,
                "device": checks["actual_model_ready"]["device"],
                "checks": len(checks),
                "report": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
