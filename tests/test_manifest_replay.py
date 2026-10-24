import json
from pathlib import Path

import pytest

from threadlens.config import Settings
from threadlens.data import replay
from threadlens.data.hn import text_hash


def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "published.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "comment_id": 123,
                        "thread_id": 100,
                        "permalink": "https://news.ycombinator.com/item?id=123",
                        "source_timestamp": "2026-01-01T00:00:00Z",
                        "collected_at": "2026-10-24T00:00:00Z",
                        "text_hash": text_hash("Complete synthetic test text."),
                        "dataset_version": "test-v1",
                    }
                ]
            }
        )
    )
    return path


def test_replay_verifies_hash_and_keeps_published_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "settings", Settings(root=tmp_path))
    monkeypatch.setattr(
        replay.HNClient,
        "item",
        lambda self, identifier: {
            "id": identifier,
            "type": "comment",
            "text": "<p>Complete synthetic test text.</p>",
        },
    )
    result = replay.replay_manifest(manifest(tmp_path))
    assert result["complete"]
    assert result["matched"] == 1
    record = json.loads((tmp_path / "data/raw/comments.jsonl").read_text())
    assert record["collected_at"] == "2026-10-24T00:00:00Z"
    assert record["text"] == "Complete synthetic test text."


@pytest.mark.parametrize(
    "source",
    [None, {"type": "comment", "deleted": True}, {"type": "comment", "text": "Source edited"}],
)
def test_changed_or_deleted_source_never_silently_replaces_benchmark(tmp_path, monkeypatch, source):
    monkeypatch.setattr(replay, "settings", Settings(root=tmp_path))
    monkeypatch.setattr(replay.HNClient, "item", lambda self, identifier: source)
    with pytest.raises(ValueError, match="changed or disappeared"):
        replay.replay_manifest(manifest(tmp_path))
    assert not (tmp_path / "data/raw/comments.jsonl").exists()
    assert not json.loads((tmp_path / ".local/replay_status.json").read_text())["complete"]
