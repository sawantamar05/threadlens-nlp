# Data card

ThreadLens uses a newly collected snapshot of 720 complete English-language Hacker News comments from 36 technical/AI discussion threads, at most 20 per thread. Collection ran on 2026-10-24. The original comments span 2013-12-02 through 2026-10-07; “fresh” describes this project's collection, not the age of the source discussions. No course data, annotations, checkpoints, or Git history were used.

## Collection and retention

Story discovery uses [HN Algolia search](https://hn.algolia.com/api) queries recorded in `reports/collection_manifest.json`. All source comments and story records are retrieved through the [official Hacker News Firebase API](https://github.com/HackerNews/API). Ranked search and the first 36 eligible groups bias this sample toward prominent AI discussions; it is not representative of all HN or all English discussion.

Each run uses four request workers, 15-second timeouts, three attempts with exponential backoff, an SQLite response cache, a seeded bounded tree traversal, exact normalized-text hash deduplication, and resumable JSONL writes. HTML becomes plain text with paragraph boundaries. Deleted, dead, missing, and empty records are filtered. `langdetect` is seeded at 42; English text with at least six whitespace-separated words is retained. This heuristic can reject valid short contributions or misidentify language. Comment length does not determine its discourse label.

The complete source text remains in ignored `data/raw/comments.jsonl`; the longest record is 3,239 characters (590 words). Public manifests contain IDs, source permalinks, source/collection timestamps, text hashes, sampling configuration, dataset version, and counts. They omit comment text and author handles. Cache and annotation databases remain local. Cached snapshots preserve reproducibility; later source edits/deletions may prevent byte-identical reconstruction from the public API. Do not treat the code's MIT license as a license for source comments. The [YC terms](https://www.ycombinator.com/legal/) and API documentation were inspected; this project does not assert a general redistribution license for user submissions. Only original synthetic fixtures are distributed.

## Taxonomy

Apply this order to a whole comment:

1. **request**: a genuine primary request for information, clarification, advice, validation, or action.
2. **analysis**: otherwise, a substantive claim supported by a specific observation, relevant comparison, measurement, quotation, or explanatory mechanism.
3. **commentary**: otherwise, an opinion, reaction, joke, unsupported assertion, or resource share.

A rhetorical question can be commentary or analysis. A genuine question with technical context is still request when obtaining an answer is its primary purpose. A number or link alone does not establish analysis. Technical language and length do not determine the label. A comment can mix all three functions; selecting its primary contribution is subjective. The taxonomy measures discourse function, never factual truth.

## Weak-label provenance

All 720 benchmark labels are **weak_label / unreviewed**. There are zero independently human-labeled benchmark examples. A local NLI teacher, [`cross-encoder/nli-distilroberta-base`](https://huggingface.co/cross-encoder/nli-distilroberta-base), Apache-2.0, revision `b14d131f9d32668a5e6a982729b57ff6ed5dfcbd`, compares the full versioned taxonomy hypotheses in `src/threadlens/labeling/weak.py`.

Complete text is divided into 384-token premise windows with 32-token overlap. Entailment logits are averaged over windows, then normalized across the three hypotheses. The teacher is an approximate discourse labeler: it is not guaranteed to enforce the taxonomy's decision order or distinguish rhetorical questions. Long comments with mixed purposes may be especially ambiguous. Its scores are uncalibrated. Label configuration, model revision, timestamps, window counts, scores, provenance, and review status are preserved locally per record; public `reports/labeling_manifest.json` records the aggregate configuration. No class counts were forced, and no labels were invented to balance classes.

| Label | All | Train | Validation | Test |
|---|---:|---:|---:|---:|
| analysis | 206 | 140 | 28 | 38 |
| request | 54 | 42 | 4 | 8 |
| commentary | 460 | 318 | 68 | 74 |
| Total | 720 | 500 | 100 | 120 |

## Frozen partitions and evaluation limits

`GroupShuffleSplit` with seed 42 creates 25/5/6 story groups in train/validation/test, respectively: 69.4%/13.9%/16.7% of records. Story groups never overlap, and normalized exact-text duplicates were removed globally. The split is frozen before baseline/transformer comparison in `reports/split_manifest.json`. Near-duplicate paraphrases and repeated subject matter across distinct threads remain possible.

Training seeds 42, 7, and 2024 vary training randomness on these same partitions. They do not measure uncertainty due to another data split. Four validation requests and eight test requests provide particularly unstable minority-class recall estimates. Reported metrics measure agreement with a fallible NLI teacher, not independently validated semantic accuracy. The teacher is not evaluated against its own generated labels as an independent baseline.

## Review and future versions

The annotation UI stores original drafts separately from append-only revisions. Accepting or editing an AI suggestion is `ai_assisted_reviewed`, not independent annotation. Blind first labels hide suggestions; only a first blind human submission can have `human_independent` provenance. Review state is local and is not automatically folded into the frozen benchmark. A later human benchmark requires a new explicitly versioned dataset/split and a separate evaluation.

`data/demo/` contains original synthetic examples with `synthetic_demo` provenance. They support offline application demonstrations and never contribute to the real benchmark.
