# Data boundaries

`demo/comments.csv` contains original synthetic examples authored for this project and distributed under the repository's MIT license. It is an offline functional demonstration, never part of the Hacker News benchmark.

Fresh Hacker News comment text, labels, frozen datasets, HTTP caches, row-level predictions, and SQLite review state stay under ignored `.local/` or `data/raw/`. Model weights stay under ignored `artifacts/`. Public reports contain aggregate metrics and source IDs only. Do not add raw comments or reviewer notes to Git.

The source API exposes public comments but does not provide a blanket redistribution license for all user-authored text. Fetch source identifiers through the collector for local research; later edits and deletions can prevent exact reconstruction. See `docs/data_card.md` and `docs/third_party.md`.
