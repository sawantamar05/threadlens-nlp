"""Refetch published source IDs without pretending changed source text is identical."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from threadlens.config import settings
from threadlens.data.hn import HNClient, clean_html, text_hash


def replay_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    references = manifest.get("records", [])
    if not references or len(references) > 10000:
        raise ValueError("Manifest must contain 1–10,000 source identifiers.")
    expected_ids = [row["comment_id"] for row in references]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Manifest contains duplicate comment identifiers.")
    output = settings.root / "data/raw/comments.jsonl"
    if output.exists():
        existing = [json.loads(line) for line in output.read_text().splitlines() if line]
        if [(row["comment_id"], row["text_hash"]) for row in existing] != [
            (row["comment_id"], row["text_hash"]) for row in references
        ]:
            raise ValueError("Existing local corpus differs; use a separate THREADLENS_ROOT.")
    client = HNClient()
    records, issues = [], []
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            values = executor.map(client.item, expected_ids)
            for reference, value in zip(references, values, strict=True):
                identifier = reference["comment_id"]
                if (
                    not value
                    or value.get("deleted")
                    or value.get("dead")
                    or value.get("type") != "comment"
                ):
                    issues.append(
                        {"comment_id": identifier, "reason": "missing/deleted/dead/non-comment"}
                    )
                    continue
                text = clean_html(value.get("text", ""))
                if not text or text_hash(text) != reference["text_hash"]:
                    issues.append(
                        {"comment_id": identifier, "reason": "cleaned source hash differs"}
                    )
                    continue
                records.append(
                    {
                        "comment_id": identifier,
                        "thread_id": reference["thread_id"],
                        "story_id": reference["thread_id"],
                        "permalink": reference["permalink"],
                        "source_timestamp": reference["source_timestamp"],
                        "collected_at": reference["collected_at"],
                        "text": text,
                        "text_hash": reference["text_hash"],
                        "dataset_version": reference["dataset_version"],
                    }
                )
    finally:
        client.http.close()
    result = {
        "mode": "published-ID replay",
        "requested": len(references),
        "matched": len(records),
        "issues": issues,
        "complete": not issues,
        "snapshot_notice": "Original collection timestamps retained for manifest replay; HTTP cache records actual refetch times.",
    }
    settings.local.mkdir(parents=True, exist_ok=True)
    (settings.local / "replay_status.json").write_text(json.dumps(result, indent=2))
    if issues:
        raise ValueError(
            f"{len(issues)} source items changed or disappeared. Local replay_status.json records IDs; corpus not overwritten. Use a new dataset version for a changed benchmark."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    labeler_manifest = settings.reports / "labeling_manifest.json"
    labeler_config = settings.local / "labeler_config.json"
    if labeler_manifest.exists() and not labeler_config.exists():
        labeler_config.write_text(
            json.dumps(json.loads(labeler_manifest.read_text())["labeler"], indent=2)
        )
    return result
