# Progress and handoff

## Current state

Engineering, local verification, and public GitHub publication are complete. Repository: https://github.com/sawantamar05/threadlens-nlp. Verified initial implementation commit `f9da0772647ab7ec87c8f64e1324e06038b73973` matched local and remote main; [GitHub Actions run 40130972332](https://github.com/sawantamar05/threadlens-nlp/actions/runs/40130972332) passed lint, formatting, CPU tests and offline demo rebuild. Final selected local model: `distilbert-improved-seed 42`. Main services use loopback API 8000 / UI 8501; the isolated synthetic demo is separate. Local API, readiness, API docs and UI health returned HTTP 200. Current CI status is available from the repository Actions page.

## Completed work

- Inspected Apple M3, 16 GB RAM, 8 CPU cores, macOS 26.6.2, Python 3.11.0 and approximately 19 GiB available disk. Confirmed MPS outside the restricted sandbox. No ancestor AGENTS.md; destination was absent and outside existing Git repositories. GitHub authenticated as sawantamar05 and requested name was absent. Protected CodePath files were not read, copied or modified.
- Created an isolated environment with full and lightweight dependency locks. Implemented installable package, centralized configuration, six-screen Streamlit UI, typed FastAPI API, cached inference, SQLite collection/review state, safe CSV processing, original synthetic demo and process-safe launcher/stop commands.
- Fresh official HN retrieval: 720 complete English comments / 36 discussions / at most 20 per discussion. Full source texts stay local. Automatically generated NLI weak labels: 206 analysis / 54 request / 460 commentary; no counts forced.
- Froze 500/100/120 comments in 25/5/6 disjoint story groups. Four validation and eight test requests limit minority estimates. Implemented ID/hash replay with changed/deleted-source refusal and pinned pretrained source revisions.
- Committed preregistration before comparisons, ran majority/TF-IDF/original DistilBERT across seeds 42, 7, 2024, declared exactly one validation-selected class-weighted-loss change in a commit before improved test evaluation, then ran all three improved seeds. All 12 runs retained. Saved/reloaded tokenizer/model and verified mapping/logits.
- Selected improved configuration by validation mean macro F1; predeclared representative seed 42 is active. Test macro F1 means:majority 0.2543, TF-IDF 0.3346, original 0.2784, weighted 0.4883. Selected seed 42 macro F1=0.5003,request recall = 0.50, warm MPS model latency 12.03 ms / 20 samples. MacroF1goal 0.65 missed; independent semantic accuracy unmeasured.
- 73 tests passed with Ruff/formatting. Fresh full and torch-free environments both passed 73 tests. Actual original/selected model API integrations passed 12 checks each; selected CPU cache-only inference passed. Port conflicts, stale process identity refusal, separate CLIstop and child cleanup passed.
- Browser verified real/synthetic workflows, CSV upload/download/order/escaping, actual HN analysis and interactive filter 20 → 10, actual 12-run report download/charts/confusion, blind synthetic review save/reload, missingnetwork/model states, desktop and 390 px viewports. No JavaScript/console errors in final reports. Actual screenshots contain only original synthetic text or aggregate results.
- Completed README, architecture, data/model cards, experiments, troubleshooting, verification evidence, and LinkedIn/résumé materials. Original code is MIT; model/source attribution is separate. AI assistance is disclosed.

## Evidence and commands

- `ruff check .`; `ruff format --check .`; `pytest -q`: 73 passed.
- `threadlens collect --manifest reports/collection_manifest.json`: 720 cached source hashes matched.
- `threadlens experiments --stage initial`; committed `configs/improvement.json`; `threadlens experiments --stage improved`: all 12 measured runs.
- `python scripts/integration_check.py --model distilbert-improved-seed 42`: 12 actual API checks, MPS.
- `python scripts/browser_smoke.py --url http://127.0.0.1:8501 --prefix trained --model 'DistilBERT class-weighted' --experiments --story 47340079`.
- `python scripts/browser_error_states.py`:actual empty registry 503 and closed-API error states; owned fixture processes stopped.
- Fresh clone/fullenv/torch-freeenv, launcher shutdown and safe stop: `reports/clean_clone_checks.json`.
- Full recorded metrics/config/versions/revisions: `reports/experiments.json`; row-level text predictions remain ignored `.local/predictions/`.

## Decisions and issues resolved

API owns inference; Streamlit has no duplicate neural copy. Full source comments remain preserved despite 256-token training/inference limits. Reviews append and never rewrite the frozen weak benchmark. Synthetic fixtures stay isolated. Only representative seed 42 weights are retained; metrics and row predictions for every seed persist. An interrupted precommit exploratory execution was rerun under the committed unchanged protocol; reported metadata retains actual per-run HEAD revisions as the implementation history advanced.

Cold AppTest imports exceeded the initial 3-second test budget; 30 seconds and sequential clean-env reruns fixed it without weakening assertions. Browser custom controls and streamed charts required observed-role locators and render waits. Caption container opacity was corrected to 1 and verified in the actual browser. One benign Starlette TestClient/httpx deprecation warning remains.

## Limitations and next action

All benchmark labels are unreviewed weak labels. There is no independent human benchmark, calibrated confidence, production deployment or demonstrated user/business impact. Source edits/deletions may prevent exact replay. Minority support is small; topic/English/length sampling bias, parent-context dependence, truncation and teacher errors remain.

No engineering or publication dependency remains. The new public repository contains the source, locks, reports, docs and 19 actual screenshots; description/topics and main branch are configured. The publication scan found no credentials, raw comments, local databases, virtual environments or model weights among the tracked files. Local and remote implementation HEADs matched, required files/images were fetched through GitHub, and CI passed.

Exact next action for use: open http://127.0.0.1:8501. To restart from this checkout, run `.venv/bin/threadlens stop` then `.venv/bin/threadlens serve`. This preserves trained local artifacts and review history. Future model-quality work is a separately versioned independently blind-labeled benchmark; no further search or relabeling was used to alter these published results.
