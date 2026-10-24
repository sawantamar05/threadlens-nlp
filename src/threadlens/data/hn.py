"""Bounded, cached public Hacker News retrieval; no arbitrary URL fetching."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup

from threadlens.config import settings

API = "https://hacker-news.firebaseio.com/v0"
DATASET_VERSION = "hn-2026-10-24-v1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def clean_html(value: str) -> str:
    """Preserve paragraph and preformatted block boundaries without executing HTML."""
    soup = BeautifulSoup(value, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all("br"):
        tag.replace_with("\n")
    for tag in soup.find_all(["p", "pre", "blockquote"]):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    text = soup.get_text()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip()).encode()).hexdigest()


def parse_story_id(value: str | int) -> int:
    raw = str(value).strip()
    if raw.isdecimal():
        result = int(raw)
    else:
        parsed = urlparse(raw)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "news.ycombinator.com"
            or parsed.path != "/item"
            or parsed.fragment
        ):
            raise ValueError(
                "Use a positive Hacker News story ID or https://news.ycombinator.com/item?id=…"
            )
        query = parse_qs(parsed.query)
        if set(query) != {"id"} or len(query["id"]) != 1 or not query["id"][0].isdecimal():
            raise ValueError("The Hacker News URL must contain one numeric id parameter.")
        result = int(query["id"][0])
    if not 0 < result < 2**63:
        raise ValueError("Story ID must be a positive 64-bit integer.")
    return result


@contextmanager
def _database(path: Path):
    connection = sqlite3.connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


class HNClient:
    """Four concurrent requests, 15-second timeout, three attempts and durable item cache."""

    def __init__(self, cache_path: Path | None = None, *, offline: bool = False):
        self.path = cache_path or settings.local / "collection.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.http = httpx.Client(timeout=15, follow_redirects=False)
        with _database(self.path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS item_cache (id INTEGER PRIMARY KEY, payload TEXT, fetched_at TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS collection_runs (version TEXT PRIMARY KEY, manifest TEXT NOT NULL)"
            )

    def _get(self, url: str) -> object:
        last = None
        for attempt in range(3):
            try:
                response = self.http.get(url)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                if attempt < 2:
                    time.sleep(0.5 * 2**attempt)
        raise RuntimeError(f"Public Hacker News request failed after 3 attempts: {last}")

    def item(self, item_id: int) -> dict | None:
        item_id = parse_story_id(item_id)
        with _database(self.path) as db:
            cached = db.execute("SELECT payload FROM item_cache WHERE id=?", (item_id,)).fetchone()
        if cached:
            return json.loads(cached[0])
        if self.offline:
            raise RuntimeError(
                f"Item {item_id} is not cached. Connect to the internet to fetch it."
            )
        value = self._get(f"{API}/item/{item_id}.json")
        with _database(self.path) as db:
            db.execute(
                "INSERT OR REPLACE INTO item_cache VALUES (?,?,?)",
                (item_id, json.dumps(value), utc_now()),
            )
        return value if isinstance(value, dict) else None

    def story_comments(
        self, story_id: int, limit: int = 50, *, seed: int = 42, pool: int | None = None
    ) -> list[dict]:
        story = self.item(story_id)
        if not story or story.get("deleted") or story.get("dead") or story.get("type") != "story":
            raise ValueError("Item is missing, deleted, dead, or is not a Hacker News story.")
        if not 1 <= limit <= 100:
            raise ValueError("Comment limit must be between 1 and 100.")
        max_nodes = min(pool or limit * 5, 400)
        queue = list(story.get("kids", []))
        random.Random(seed + story_id).shuffle(queue)
        seen = set()
        records = []
        inspected = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            while queue and inspected < max_nodes:
                chunk, queue = queue[:4], queue[4:]
                fresh = [i for i in chunk if i not in seen]
                seen.update(fresh)
                for value in executor.map(self.item, fresh):
                    inspected += 1
                    if not value:
                        continue
                    queue.extend(value.get("kids", []))
                    if value.get("deleted") or value.get("dead") or value.get("type") != "comment":
                        continue
                    text = clean_html(value.get("text", ""))
                    if not text:
                        continue
                    comment_id = value["id"]
                    records.append(
                        {
                            "comment_id": comment_id,
                            "thread_id": story_id,
                            "story_id": story_id,
                            "permalink": f"https://news.ycombinator.com/item?id={comment_id}",
                            "source_timestamp": datetime.fromtimestamp(
                                value.get("time", 0), UTC
                            ).isoformat(),
                            "collected_at": utc_now(),
                            "text": text,
                            "text_hash": text_hash(text),
                            "dataset_version": DATASET_VERSION,
                        }
                    )
        random.Random(seed + story_id).shuffle(records)
        return records[:limit]


def fetch_story_comments(story_id: int, limit: int = 50) -> dict:
    client = HNClient()
    story = client.item(story_id)
    comments = client.story_comments(story_id, limit)
    return {
        "story_id": story_id,
        "title": clean_html((story or {}).get("title", "")),
        "comments": comments,
        "warnings": [
            "Cached items are snapshots; later edits and deletions may differ from the live source."
        ]
        + (
            [f"Only {len(comments)} usable comments were found within the bounded traversal."]
            if len(comments) < limit
            else []
        ),
    }


def discover_stories(client: HNClient) -> list[int]:
    """Use HN's documented Algolia search for discovery; fetch source text only from official API."""
    result = []
    for query in ["AI", "database", "Python", "compiler", "Postgres", "LLM", "Linux", "OpenAI"]:
        url = str(
            httpx.URL(
                "https://hn.algolia.com/api/v1/search",
                params={
                    "query": query,
                    "tags": "story",
                    "numericFilters": "num_comments>=40",
                    "hitsPerPage": 60,
                },
            )
        )
        for hit in client._get(url).get("hits", []):
            identifier = int(hit["objectID"])
            if identifier not in result:
                result.append(identifier)
    return result


def collect_dataset(*, threads: int = 36, per_thread: int = 20, seed: int = 42) -> dict:
    """Resume deterministic selection from cached snapshots, retaining full cleaned text locally."""
    from langdetect import DetectorFactory, LangDetectException, detect

    if threads < 1 or not 1 <= per_thread <= 20:
        raise ValueError("Use positive thread count and at most 20 comments per thread.")
    DetectorFactory.seed = seed
    output = settings.root / "data/raw/comments.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    ids_path = settings.local / "candidate_stories.json"
    client = HNClient()
    if ids_path.exists():
        candidates = json.loads(ids_path.read_text())
    else:
        candidates = discover_stories(client)
        ids_path.write_text(json.dumps(candidates))
    records = []
    hashes = set()
    counts = {}
    rejected = {"non_english_or_short": 0, "duplicate": 0, "unavailable_thread": 0}
    existing = (
        [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    )
    for record in existing:
        if record["text_hash"] not in hashes:
            records.append(record)
            hashes.add(record["text_hash"])
            counts[record["thread_id"]] = counts.get(record["thread_id"], 0) + 1
    for story_id in candidates:
        if len(counts) >= threads:
            break
        if story_id in counts:
            continue
        try:
            pool = client.story_comments(story_id, 100, pool=160, seed=seed)
        except (ValueError, RuntimeError):
            rejected["unavailable_thread"] += 1
            continue
        chosen = []
        for record in pool:
            if record["text_hash"] in hashes:
                rejected["duplicate"] += 1
                continue
            try:
                english = len(record["text"].split()) >= 6 and detect(record["text"]) == "en"
            except LangDetectException:
                english = False
            if not english:
                rejected["non_english_or_short"] += 1
                continue
            chosen.append(record)
            hashes.add(record["text_hash"])
            if len(chosen) == per_thread:
                break
        if len(chosen) < per_thread:
            # Avoid partial groups in this frozen benchmark; cache still allows later retrieval.
            hashes.difference_update(r["text_hash"] for r in chosen)
            continue
        records.extend(chosen)
        counts[story_id] = len(chosen)
        output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        print(f"Collected {len(records)} comments from {len(counts)} threads", flush=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else ""
    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at": utc_now(),
        "comment_count": len(records),
        "thread_count": len(counts),
        "thread_counts": counts,
        "dataset_sha256": digest,
        "sampling": {
            "seed": seed,
            "target_threads": threads,
            "max_per_thread": per_thread,
            "traversal_pool": 160,
            "candidate_selection": "Algolia ranked technical query stories with >=40 comments",
            "queries": [
                "AI",
                "database",
                "Python",
                "compiler",
                "Postgres",
                "LLM",
                "Linux",
                "OpenAI",
            ],
            "english_filter": "langdetect seed42, English, >=6 whitespace-separated words",
        },
        "rejected": rejected,
        "source_api": API,
        "raw_text_published": False,
        "records": [
            {
                key: r[key]
                for key in (
                    "comment_id",
                    "thread_id",
                    "permalink",
                    "source_timestamp",
                    "collected_at",
                    "text_hash",
                    "dataset_version",
                )
            }
            for r in records
        ],
        "limitation": "Cached snapshot; edits/deletions can prevent byte-identical recollection. Short comments and language detection introduce sampling bias.",
        "target_met": len(records) >= 600 and len(counts) >= 30,
    }
    settings.reports.mkdir(parents=True, exist_ok=True)
    (settings.reports / "collection_manifest.json").write_text(json.dumps(manifest, indent=2))
    with _database(client.path) as db:
        db.execute(
            "INSERT OR REPLACE INTO collection_runs VALUES (?,?)",
            (DATASET_VERSION, json.dumps(manifest)),
        )
    return manifest
