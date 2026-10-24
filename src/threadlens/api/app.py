"""Loopback-first FastAPI service; all clients share a single inference owner."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from threadlens.api.middleware import RequestBodyLimitMiddleware
from threadlens.api.schemas import (
    AnnotationRequest,
    AnnotationResponse,
    BatchRequest,
    BatchResponse,
    ModelsResponse,
    Prediction,
    PredictionRequest,
    ThreadRequest,
    ThreadResponse,
)
from threadlens.config import LABELS, Settings
from threadlens.config import settings as default_settings
from threadlens.models.inference import ModelManager, ModelUnavailable
from threadlens.storage.annotations import AnnotationStore
from threadlens.validation import InputInvalid, has_visible_text

logger = logging.getLogger(__name__)


def create_app(config: Settings | None = None, *, settings: Settings | None = None) -> FastAPI:
    config = settings or config or default_settings
    manager = ModelManager(config)
    store = AnnotationStore(config.database)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        labels_path = config.local / "labeled.jsonl"
        if labels_path.is_file():
            try:
                records = [
                    json.loads(line) for line in labels_path.read_text().splitlines() if line
                ]
                store.import_records(records)
            except (ValueError, OSError) as exc:
                logger.error("Could not import annotation queue: %s", exc)
        yield

    application = FastAPI(
        title="ThreadLens — NLP Discussion Intelligence",
        version="0.1.0",
        description=(
            "Local discourse classification: analysis, request, commentary. "
            "Model scores are uncalibrated. Weak-label evaluation measures agreement, "
            "not independently validated semantic accuracy."
        ),
        lifespan=lifespan,
    )
    application.state.manager = manager
    application.state.store = store
    application.state.settings = config

    @application.exception_handler(ModelUnavailable)
    async def model_unavailable(_: Request, exc: ModelUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc), "ready": False})

    @application.exception_handler(InputInvalid)
    async def invalid_service_input(_: Request, exc: InputInvalid):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    application.add_middleware(RequestBodyLimitMiddleware, max_bytes=config.max_csv_bytes)

    @application.exception_handler(RequestValidationError)
    async def invalid_input(_: Request, exc: RequestValidationError):
        # Pydantic normally echoes rejected input, including complete private comments.
        errors = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.get("/health", tags=["Operations"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "threadlens", "version": "0.1.0"}

    @application.get("/ready", tags=["Operations"])
    def ready(model_id: str | None = Query(default=None, max_length=128)) -> dict[str, Any]:
        return manager.ensure_ready(model_id)

    @application.get("/models", response_model=ModelsResponse, tags=["Models"])
    def models():
        return manager.models()

    @application.post("/predict", response_model=Prediction, tags=["Prediction"])
    def predict(request: PredictionRequest):
        result = manager.predict([request.text], request.model_id)[0]
        store.mark_text_exposed([request.text])
        return result

    @application.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
    def predict_batch(request: BatchRequest):
        start = time.perf_counter()
        results = manager.predict(request.texts, request.model_id)
        store.mark_text_exposed(request.texts)
        return {"results": results, "processing_ms": round((time.perf_counter() - start) * 1000, 3)}

    @application.post("/analyze-thread", response_model=ThreadResponse, tags=["Threads"])
    def analyze_thread(request: ThreadRequest):
        from threadlens.data.hn import fetch_story_comments, parse_story_id

        # Check readiness before making public network requests.
        manager.ensure_ready(request.model_id)
        try:
            result = fetch_story_comments(parse_story_id(request.story), request.limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail="Hacker News retrieval timed out. Retry with a smaller comment limit.",
            ) from exc
        except (RuntimeError, httpx.HTTPError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Hacker News is unavailable: {exc}. Retry later or use cached data offline.",
            ) from exc
        valid = []
        too_long = 0
        seen = set()
        for comment in result["comments"]:
            if not has_visible_text(comment.get("text", "")) or "\x00" in comment["text"]:
                continue
            if len(comment["text"]) > config.max_chars:
                too_long += 1
                continue
            if comment["comment_id"] not in seen:
                valid.append(comment)
                seen.add(comment["comment_id"])
        predictions = (
            manager.predict([comment["text"] for comment in valid], request.model_id)
            if valid
            else []
        )
        store.mark_text_exposed([comment["text"] for comment in valid])
        comments = []
        for source, prediction in zip(valid, predictions, strict=True):
            reviewed_label = None
            reviewed_provenance = None
            try:
                history = store.history(str(source["comment_id"]), blind=True)
                if history["revisions"]:
                    reviewed_label = history["revisions"][-1]["label"]
                    reviewed_provenance = history["revisions"][-1]["provenance"]
            except KeyError:
                pass
            comments.append(
                {
                    **source,
                    **prediction,
                    "reviewed_label": reviewed_label,
                    "reviewed_provenance": reviewed_provenance,
                }
            )
        counts = Counter(comment["label"] for comment in comments)
        warnings = list(result.get("warnings", []))
        if too_long:
            warnings.append(
                f"Skipped {too_long} complete comments exceeding the {config.max_chars}-character service bound."
            )
        if not comments:
            warnings.append("No usable comments were available within this bounded traversal.")
        return {
            "story_id": result["story_id"],
            "title": result["title"],
            "comments": comments,
            "counts": {label: counts[label] for label in LABELS},
            "proportions": {
                label: counts[label] / len(comments) if comments else 0 for label in LABELS
            },
            "warnings": warnings,
        }

    @application.get("/experiments", tags=["Experiments"])
    def experiments() -> dict[str, Any]:
        path = config.reports / "experiments.json"
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="No experiment report. Run 'threadlens experiments' to generate one.",
            )
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=503, detail="Experiment report is unreadable; regenerate it."
            ) from exc

    @application.get("/review/next", tags=["Annotation review"])
    def review_next(blind: bool = True) -> dict[str, Any]:
        return {"item": store.next_item(blind)}

    @application.get("/review/status", tags=["Annotation review"])
    def review_status() -> dict[str, int]:
        return store.counts()

    @application.get("/annotations/{comment_id}", tags=["Annotation review"])
    def annotation_history(comment_id: str, blind: bool = True) -> dict[str, Any]:
        try:
            return store.history(comment_id, blind)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Comment is not in the review queue."
            ) from exc

    @application.post("/annotations", response_model=AnnotationResponse, tags=["Annotation review"])
    def annotate(request: AnnotationRequest):
        try:
            return store.save(**request.model_dump())
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Comment is not in the review queue."
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return application


app = create_app()
