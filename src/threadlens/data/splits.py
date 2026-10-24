"""Freeze story-group partitions before model comparisons."""

import hashlib
import json
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from threadlens.config import LABELS, settings


def create_splits(records: list[dict], seed: int = 42) -> dict:
    groups = np.array([r["thread_id"] for r in records])
    if len(set(groups)) < 10:
        raise ValueError(
            "At least ten discussion groups are required for a useful three-way split."
        )
    indices = np.arange(len(records))
    train, remainder = next(
        GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed).split(
            indices, groups=groups
        )
    )
    val_relative, test_relative = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(
            remainder, groups=groups[remainder]
        )
    )
    partitions = {
        "train": train.tolist(),
        "validation": remainder[val_relative].tolist(),
        "test": remainder[test_relative].tolist(),
    }
    split = {
        "seed": seed,
        "method": "GroupShuffleSplit 70/15/15; fixed threads for every training seed",
        "dataset_hash": hashlib.sha256(
            json.dumps(
                [(r["comment_id"], r["text_hash"], r["label"]) for r in records],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "partitions": {},
    }
    for name, positions in partitions.items():
        subset = [records[i] for i in positions]
        counts = dict(Counter(r["label"] for r in subset))
        if set(counts) != set(LABELS):
            raise ValueError(
                f"Frozen {name} lacks a taxonomy class: {counts}. Collect more data rather than fabricate labels."
            )
        split["partitions"][name] = {
            "indices": positions,
            "comment_ids": [r["comment_id"] for r in subset],
            "thread_ids": sorted(set(r["thread_id"] for r in subset)),
            "count": len(subset),
            "label_counts": counts,
        }
    sets = [set(part["thread_ids"]) for part in split["partitions"].values()]
    if any(sets[a] & sets[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
        raise AssertionError("Discussion leakage detected.")
    return split


def freeze_splits(records: list[dict]) -> dict:
    path = settings.reports / "split_manifest.json"
    result = create_splits(records)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != result:
            raise ValueError(
                "Existing split manifest differs. Use a new dataset version for a new benchmark."
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result
