"""Process-local inference registry. Never substitutes a missing trained artifact."""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from threadlens.config import LABELS, Settings, settings
from threadlens.validation import validate_texts

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """An artifact is absent, incompatible, or not registered."""


class ModelManager:
    def __init__(self, config: Settings = settings):
        self.config = config
        self._loaded: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _registry(self) -> dict[str, Any]:
        path = self.config.artifacts / "model_registry.json"
        if not path.is_file():
            return {"active_model": None, "models": []}
        try:
            registry = json.loads(path.read_text())
            if not isinstance(registry, dict) or not isinstance(registry.get("models"), list):
                raise ValueError("models must be a list")
            if registry.get("active_model") is not None and not isinstance(
                registry["active_model"], str
            ):
                raise ValueError("active_model must be a model identifier or null.")
            identifiers = set()
            for entry in registry["models"]:
                if not isinstance(entry, dict) or not all(
                    isinstance(entry.get(key), str) and entry[key] for key in ("id", "path")
                ):
                    raise ValueError("Each registered model needs a nonempty id and path.")
                if entry["id"] in identifiers:
                    raise ValueError("Model identifiers must be unique.")
                identifiers.add(entry["id"])
                if any(
                    key in entry and not isinstance(entry[key], str)
                    for key in ("version", "name", "provenance")
                ):
                    raise ValueError("Model version, name, and provenance must be strings.")
                if entry.get("kind", "sklearn") not in {
                    "sklearn",
                    "majority",
                    "tfidf",
                    "transformer",
                }:
                    raise ValueError("Unsupported model kind.")
                if not isinstance(entry.get("metadata", {}), dict):
                    raise ValueError("Model metadata must be an object.")
            return registry
        except (ValueError, OSError) as exc:
            raise ModelUnavailable(f"Model registry is invalid: {exc}") from exc

    def artifact_path(self, entry: dict[str, Any]) -> Path:
        raw = Path(entry["path"])
        if raw.is_absolute():
            candidate = raw.resolve()
        elif raw.parts and raw.parts[0] == "artifacts":
            candidate = (self.config.root / raw).resolve()
        else:
            candidate = (self.config.artifacts / raw).resolve()
        if not candidate.is_relative_to(self.config.artifacts.resolve()):
            raise ModelUnavailable(
                "Artifact path must remain inside the project artifacts directory."
            )
        return candidate

    def _metadata(self, entry: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(entry.get("metadata", {}))
        if entry.get("kind") == "transformer":
            length = metadata.get("max_length")
            if length is None:
                training = metadata.get("training_config", {})
                if not isinstance(training, dict):
                    raise ModelUnavailable("Model training_config metadata must be an object.")
                length = training.get("max_length", 256)
            if not isinstance(length, int) or isinstance(length, bool) or not 8 <= length <= 512:
                raise ModelUnavailable(
                    "DistilBERT maximum input length must be between 8 and 512 tokens."
                )
            metadata["max_length"] = length
            metadata["input_policy"] = "Leading token window; full source text remains unchanged."
        else:
            metadata["max_length"] = None
            metadata["input_policy"] = "Complete bounded input; no token-window truncation."
        metadata["inference_device"] = self._loaded.get(entry["id"], {}).get("device", "not_loaded")
        return metadata

    def _artifact_signature(self, entry: dict[str, Any]) -> tuple:
        path = self.artifact_path(entry)
        if path.is_file() or path.suffix == ".joblib":
            files = [path]
        elif entry.get("kind") == "transformer":
            files = [
                path / name
                for name in (
                    "config.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
            ]
        else:
            files = [path / "model.joblib"]
        stats = tuple(
            (str(file), file.stat().st_size, file.stat().st_mtime_ns)
            for file in files
            if file.is_file()
        )
        return (entry.get("version"), self._metadata(entry)["max_length"], stats)

    def models(self) -> dict[str, Any]:
        registry = self._registry()
        entries = []
        for entry in registry["models"]:
            path = self.artifact_path(entry)
            kind = entry.get("kind", "sklearn")
            available = (
                path.is_file()
                if path.suffix == ".joblib"
                else (
                    (path / "config.json").is_file()
                    and ((path / "tokenizer.json").is_file() or (path / "vocab.txt").is_file())
                    and (
                        (path / "model.safetensors").is_file()
                        or (path / "pytorch_model.bin").is_file()
                    )
                    if kind == "transformer"
                    else (path / "model.joblib").is_file()
                )
            )
            entries.append(
                {
                    "id": entry["id"],
                    "name": entry.get("name", entry["id"]),
                    "kind": kind,
                    "version": entry.get("version", "unversioned"),
                    "provenance": entry.get("provenance", "unknown"),
                    "available": available,
                    "metadata": self._metadata(entry),
                }
            )
        return {"active_model": registry.get("active_model"), "models": entries}

    def _entry(self, model_id: str | None) -> dict[str, Any]:
        registry = self._registry()
        selected = model_id or registry.get("active_model")
        for entry in registry["models"]:
            if entry["id"] == selected:
                return entry
        raise ModelUnavailable(
            f"Model {selected or '(active model)'} is unavailable. "
            "Run 'threadlens demo' for the offline synthetic demonstration, or "
            "'threadlens train-baselines' / 'threadlens experiments' to build real artifacts."
        )

    def _load(self, entry: dict[str, Any]) -> Any:
        model_id = entry["id"]
        signature = self._artifact_signature(entry)
        if (
            model_id in self._loaded
            and self._loaded[model_id].get("artifact_signature") == signature
        ):
            return self._loaded[model_id]
        path = self.artifact_path(entry)
        if entry.get("kind") == "transformer":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            requested = os.getenv("THREADLENS_DEVICE", "auto")
            available = torch.backends.mps.is_available()
            device = "mps" if requested == "auto" and available else requested
            if device == "auto":
                device = "cpu"
            warnings = []
            if requested in {"auto", "mps"} and not available:
                device = "cpu"
                warnings.append("MPS is unavailable; inference is running on CPU.")
            if device not in {"cpu", "mps"}:
                raise ModelUnavailable("THREADLENS_DEVICE must be auto, cpu, or mps.")
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
            actual = {int(k): v for k, v in model.config.id2label.items()}
            if actual != dict(enumerate(LABELS)) or model.config.label2id != {
                label: i for i, label in enumerate(LABELS)
            }:
                raise ModelUnavailable(f"Saved label mapping is incompatible: {actual}")
            model.to(device).eval()
            loaded = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "warnings": warnings,
            }
        else:
            file = path if path.suffix == ".joblib" else path / "model.joblib"
            if not file.is_file():
                raise ModelUnavailable(
                    f"Missing artifact for {model_id}; rebuild this model locally."
                )
            try:
                model = joblib.load(file)
            except Exception as exc:
                raise ModelUnavailable(
                    f"Cannot deserialize artifact for {model_id}; rebuild it locally."
                ) from exc
            if not callable(getattr(model, "predict_proba", None)) or not hasattr(
                model, "classes_"
            ):
                raise ModelUnavailable(
                    f"{model_id} is not a fitted probabilistic classifier; rebuild it."
                )
            classifier = getattr(model, "named_steps", {}).get("classifier", model)
            warnings = []
            if classifier.__class__.__name__ == "DummyClassifier":
                warnings.append(
                    "Majority baseline scores are degenerate; it always predicts the training majority class."
                )
            loaded = {"model": model, "device": "cpu", "warnings": warnings}
        loaded["artifact_signature"] = signature
        self._loaded[model_id] = loaded
        logger.info("Loaded inference model %s on %s", model_id, loaded["device"])
        return loaded

    def ensure_ready(self, model_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            entry = self._entry(model_id)
            try:
                loaded = self._load(entry)
            except ModelUnavailable:
                raise
            except (
                OSError,
                ValueError,
                RuntimeError,
                ImportError,
                EOFError,
                AttributeError,
                TypeError,
                pickle.UnpicklingError,
            ) as exc:
                raise ModelUnavailable(
                    f"Cannot load {entry['id']}: {exc}. Rebuild the artifact."
                ) from exc
            return {
                "ready": True,
                "model_id": entry["id"],
                "model_version": entry.get("version", "unversioned"),
                "device": loaded["device"],
                "warnings": loaded["warnings"],
            }

    def predict(self, texts: list[str], model_id: str | None = None) -> list[dict[str, Any]]:
        validate_texts(texts, self.config)
        started = time.perf_counter()
        with self._lock:
            entry = self._entry(model_id)
            try:
                loaded = self._load(entry)
                if entry.get("kind") == "transformer":
                    scores, truncated = self._neural(texts, entry, loaded)
                else:
                    raw = loaded["model"].predict_proba(texts)
                    scores = np.zeros((len(texts), len(LABELS)), dtype=float)
                    classes = loaded["model"].classes_
                    raw = np.asarray(raw, dtype=float)
                    if raw.shape != (len(texts), len(classes)):
                        raise ModelUnavailable(
                            "The classifier returned scores with the wrong shape."
                        )
                    targets = set()
                    for column, label in enumerate(classes):
                        target = (
                            int(label)
                            if isinstance(label, (int, np.integer))
                            else LABELS.index(str(label))
                        )
                        if not 0 <= target < len(LABELS) or target in targets:
                            raise ModelUnavailable(
                                "Saved classifier has an incompatible or duplicate label mapping."
                            )
                        targets.add(target)
                        scores[:, target] = raw[:, column]
                    truncated = [False] * len(texts)
            except ModelUnavailable:
                raise
            except (
                OSError,
                ValueError,
                RuntimeError,
                ImportError,
                EOFError,
                AttributeError,
                TypeError,
                pickle.UnpicklingError,
            ) as exc:
                raise ModelUnavailable(f"Prediction with {entry['id']} failed: {exc}") from exc
            scores = np.asarray(scores, dtype=float)
            if (
                scores.shape != (len(texts), len(LABELS))
                or not np.isfinite(scores).all()
                or np.any(scores < 0)
                or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)
            ):
                raise ModelUnavailable(
                    "The classifier returned invalid per-label scores; rebuild the artifact."
                )
            elapsed = (time.perf_counter() - started) * 1000
            output = []
            for row, was_truncated in zip(scores, truncated, strict=True):
                warnings = list(loaded["warnings"])
                warnings.append(
                    "Scores are uncalibrated model outputs, not probabilities of correctness."
                )
                if entry.get("provenance") == "weak_label":
                    warnings.append(
                        "Trained on weak labels; semantic accuracy is not independently validated."
                    )
                if entry.get("provenance") == "synthetic_demo":
                    warnings.append("Synthetic demonstration model; not a real benchmark result.")
                if was_truncated:
                    warnings.append(
                        "The model saw only the leading token window; later requests or evidence may be missed."
                    )
                low = float(max(row)) < self.config.low_confidence
                if low:
                    warnings.append(
                        f"Low confidence: maximum score below {self.config.low_confidence:.2f}."
                    )
                output.append(
                    {
                        "label": LABELS[int(np.argmax(row))],
                        "scores": {label: float(row[i]) for i, label in enumerate(LABELS)},
                        "model_id": entry["id"],
                        "model_version": entry.get("version", "unversioned"),
                        "truncated": bool(was_truncated),
                        "processing_ms": round(elapsed / len(texts), 3),
                        "warnings": warnings,
                        "low_confidence": low,
                    }
                )
            return output

    def _neural(self, texts: list[str], entry: dict[str, Any], loaded: dict[str, Any]):
        import torch

        tokenizer = loaded["tokenizer"]
        max_length = self._metadata(entry)["max_length"]
        lengths = tokenizer(texts, truncation=False, padding=False, return_length=True)["length"]
        rows = []
        for start in range(0, len(texts), 16):
            encoded = tokenizer(
                texts[start : start + 16],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(loaded["device"])
            try:
                with torch.inference_mode():
                    logits = loaded["model"](**encoded).logits
            except RuntimeError:
                if loaded["device"] != "mps":
                    raise
                logger.exception("MPS prediction failed, retrying on CPU")
                loaded["device"] = "cpu"
                loaded["warnings"].append("MPS operation failed; inference fell back to CPU.")
                loaded["model"].to("cpu")
                encoded = encoded.to("cpu")
                with torch.inference_mode():
                    logits = loaded["model"](**encoded).logits
            rows.extend(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.array(rows), [length > max_length for length in lengths]
