"""Append-only annotation revisions. Suggestions never become independent labels."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from threadlens.config import LABELS

PROVENANCES = {"weak_label", "ai_assisted_reviewed", "human_independent", "synthetic_demo"}


class AnnotationStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS review_items (
                  comment_id TEXT PRIMARY KEY, text TEXT NOT NULL, thread_id TEXT,
                  permalink TEXT, draft_json TEXT NOT NULL, imported_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_items_text ON review_items(text);
                CREATE TABLE IF NOT EXISTS annotation_revisions (
                  revision INTEGER PRIMARY KEY AUTOINCREMENT,
                  comment_id TEXT NOT NULL REFERENCES review_items(comment_id),
                  label TEXT NOT NULL, provenance TEXT NOT NULL,
                  review_status TEXT NOT NULL, timestamp TEXT NOT NULL,
                  notes TEXT NOT NULL, blind INTEGER NOT NULL,
                  independent_initial INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS suggestion_exposures (
                  comment_id TEXT PRIMARY KEY REFERENCES review_items(comment_id),
                  timestamp TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def import_records(self, records: list[dict[str, Any]]) -> int:
        """INSERT OR IGNORE preserves an initial draft, even after reimport/relabeling."""
        inserted = 0
        with self.connect() as db:
            for record in records:
                comment_id = str(record.get("comment_id", record.get("id", "")))
                if not comment_id or not record.get("text", "").strip():
                    continue
                draft = {
                    "label": record.get("label", record.get("draft_label")),
                    "scores": record.get("scores", record.get("draft_scores", {})),
                    "provenance": record.get("provenance", "weak_label"),
                    "review_status": record.get("review_status", "unreviewed"),
                    "labeler": record.get("labeler", record.get("labeler_model")),
                    "labeler_revision": record.get("labeler_revision"),
                    "timestamp": record.get("timestamp", record.get("labeled_at")),
                    "metadata": record.get("labeling_metadata", {}),
                }
                if draft["provenance"] not in PROVENANCES:
                    raise ValueError("Unsupported source provenance.")
                cursor = db.execute(
                    "INSERT OR IGNORE INTO review_items VALUES(?,?,?,?,?,?)",
                    (
                        comment_id,
                        record["text"],
                        str(record.get("thread_id", record.get("story_id", ""))),
                        record.get("permalink", record.get("source_permalink", "")),
                        json.dumps(draft),
                        datetime.now(UTC).isoformat(),
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def _revisions(self, db: sqlite3.Connection, comment_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in db.execute(
                "SELECT * FROM annotation_revisions WHERE comment_id=? ORDER BY revision",
                (comment_id,),
            )
        ]

    def next_item(self, blind: bool = True) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("""
                SELECT i.* FROM review_items i
                LEFT JOIN annotation_revisions r ON i.comment_id=r.comment_id
                GROUP BY i.comment_id
                ORDER BY COUNT(r.revision), i.imported_at, i.comment_id LIMIT 1
            """).fetchone()
            if row is None:
                return None
            revisions = self._revisions(db, row["comment_id"])
            show_draft = not blind or bool(revisions)
            if show_draft:
                db.execute(
                    "INSERT OR IGNORE INTO suggestion_exposures VALUES(?,?)",
                    (row["comment_id"], datetime.now(UTC).isoformat()),
                )
            draft = json.loads(row["draft_json"])
            return {
                "comment_id": row["comment_id"],
                "text": row["text"],
                "thread_id": row["thread_id"],
                "permalink": row["permalink"],
                "draft_label": draft["label"] if show_draft else None,
                "draft_scores": draft["scores"] if show_draft else None,
                "draft_provenance": draft["provenance"] if show_draft else None,
                "latest_annotation": revisions[-1] if revisions else None,
                "revision_count": len(revisions),
            }

    def history(self, comment_id: str, blind: bool = True) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM review_items WHERE comment_id=?", (comment_id,)
            ).fetchone()
            if row is None:
                raise KeyError(comment_id)
            revisions = self._revisions(db, comment_id)
            show_draft = not blind or bool(revisions)
            if show_draft:
                db.execute(
                    "INSERT OR IGNORE INTO suggestion_exposures VALUES(?,?)",
                    (comment_id, datetime.now(UTC).isoformat()),
                )
            return {
                "comment_id": comment_id,
                "draft": json.loads(row["draft_json"]) if show_draft else None,
                "revisions": revisions,
            }

    def save(
        self,
        comment_id: str,
        label: str,
        provenance: str,
        blind: bool = False,
        notes: str = "",
    ) -> dict[str, Any]:
        if label not in LABELS:
            raise ValueError("Unsupported annotation label.")
        if provenance not in {"human_independent", "ai_assisted_reviewed"}:
            raise ValueError("Manual review requires human_independent or ai_assisted_reviewed.")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            item = db.execute(
                "SELECT draft_json FROM review_items WHERE comment_id=?", (comment_id,)
            ).fetchone()
            if item is None:
                raise KeyError(comment_id)
            revisions = self._revisions(db, comment_id)
            exposed = db.execute(
                "SELECT 1 FROM suggestion_exposures WHERE comment_id=?", (comment_id,)
            ).fetchone()
            draft = json.loads(item["draft_json"])
            # Synthetic fixtures remain synthetic, regardless of the UI selection.
            if draft["provenance"] == "synthetic_demo":
                actual_provenance = "synthetic_demo"
                independent = False
            else:
                eligible_independent = bool(blind and not exposed and not revisions)
                independent = eligible_independent and provenance == "human_independent"
                if provenance == "human_independent" and not eligible_independent:
                    raise ValueError(
                        "Independent annotation requires an initial blind label before any "
                        "suggestion was displayed. Choose ai_assisted_reviewed for this revision."
                    )
                actual_provenance = provenance
            timestamp = datetime.now(UTC).isoformat()
            cursor = db.execute(
                "INSERT INTO annotation_revisions "
                "(comment_id,label,provenance,review_status,timestamp,notes,blind,independent_initial) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    comment_id,
                    label,
                    actual_provenance,
                    "reviewed",
                    timestamp,
                    notes,
                    int(blind),
                    int(independent),
                ),
            )
            return dict(
                db.execute(
                    "SELECT * FROM annotation_revisions WHERE revision=?", (cursor.lastrowid,)
                ).fetchone()
            )

    def mark_text_exposed(self, texts: list[str]) -> None:
        """Classifying queued text elsewhere also prevents later independence claims."""
        if not texts:
            return
        timestamp = datetime.now(UTC).isoformat()
        with self.connect() as db:
            for text in set(texts):
                db.execute(
                    "INSERT OR IGNORE INTO suggestion_exposures(comment_id,timestamp) "
                    "SELECT comment_id,? FROM review_items WHERE text=?",
                    (timestamp, text),
                )

    def counts(self) -> dict[str, int]:
        with self.connect() as db:
            return {
                "items": db.execute("SELECT COUNT(*) FROM review_items").fetchone()[0],
                "revisions": db.execute("SELECT COUNT(*) FROM annotation_revisions").fetchone()[0],
                "human_independent": db.execute(
                    "SELECT COUNT(*) FROM annotation_revisions WHERE provenance='human_independent'"
                ).fetchone()[0],
            }
