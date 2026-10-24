# Actual verification

The user story is: browser input → Streamlit HTTP client → typed FastAPI routes → local inference, public source retrieval or SQLite → rendered results and downloadable reports. Checks below were actually executed on the inspected Mac; The public GitHub Actions run also passed lint, formatting, CPU tests and offline demo rebuild; see the publication evidence below.

| Boundary / workflow | Evidence |
|---|---|
| Environment and dependency installation | Fresh Python 3.11 environment installed the full 73-package lock. A second environment installed the 58-package lightweight lock with torch absent. |
| Local tests | 73 passed; Ruff and formatting passed. Ordinary tests use synthetic fixtures and mocked network/model boundaries; no pretrained download is required. |
| Fresh-clone tests | Both full and torch-free environments passed 73 tests at source revision `4a505ab`. See `reports/clean_clone_checks.json`. |
| Actual trained model / API | 12 integration checks passed for original and selected class-weighted DistilBERT on MPS: readiness, normalized named scores, bounds, real truncation, missing artifacts, batch order, official HN source analysis, AI-assisted revision persistence, report provenance, OpenAPI. See `reports/local_integration*.json`. |
| Offline CPU | Saved selected tokenizer/model reloaded with network disabled; CPU readiness, batched prediction, truncation and reuse passed. See `reports/cpu_inference_check.json`. |
| Browser / real model | Prediction, model selection, empty/invalid input, 390px layout, real HN retrieval, interactive filtering from 20 to 10 displayed comments, CSV upload/inference/download, 12-run report download, charts/confusion counts, and model-information screen passed. |
| Browser / synthetic review | The synthetic demo hid drafts, saved a review with synthetic provenance, and reloaded immutable drafts plus revisions. Agent-driven checks never create human-independent labels. |
| Browser / error states | Empty registry produced actual 503 readiness, rebuild instructions and disabled inference. A UI pointed at a closed API port displayed connection instructions and disabled inference. |
| Launcher lifecycle | Offline demo API/UI health and model readiness passed. Occupied-port refusal left the original services healthy. SIGTERM closed only owned children. Separate `threadlens stop` verified launcher identity/start time, refused a mismatched record without affecting services, stopped the matching launcher, removed its record and closed both ports. |
| Source-ID reproduction | All 720 published source identifiers matched cached complete cleaned-text hashes through `collect --manifest`. Changed/deleted-source refusal has regression tests; later live changes can prevent exact replay. |
| Privacy boundaries | Raw comments, model weights, caches, reviewer notes, databases and virtual environments are ignored. Public screenshots contain synthetic prose or aggregate reports, with no raw HN comments. |

Screenshots and browser evidence live in `docs/images/` and `reports/browser_*_checks.json`. The optional browser scripts use Playwright 1.64.1 and a real Chromium instance. API/UI scores and benchmarks are actual outputs, not hardcoded fixtures masquerading as trained predictions.

## Issues found and repaired

- Fresh cold imports exceeded Streamlit AppTest's default three-second budget. The harness now allows 30 seconds; assertions and production request timeouts were retained. Both environments passed after sequential reruns. The initial failure remains recorded in the clean-clone report.
- Streamlit custom radio/toggle controls and asynchronous charts required visible-label/observed-role locators and explicit render waits in the browser harness. Final reports record the completed successful flows.
- Caption text originally inherited 0.6 container opacity. The final style sets the specified readable color and full opacity; the browser checks the actual computed color and container opacity.
- Header-only request bounds were replaced with actual streamed-byte counting; sensitive rejected input is omitted from validation responses.
- Cache-invalidating artifact checks, both label mappings, SQLite closure, score validation and cross-screen suggestion exposure were hardened with regression tests.

A benign upstream Starlette TestClient/httpx deprecation warning remains in local tests; it does not affect the checked behavior. There is no independent human benchmark. The preregistered macro F1 target was missed, and no additional search was run to conceal that result.

## Publication evidence

New public repository: [sawantamar05/threadlens-nlp](https://github.com/sawantamar05/threadlens-nlp), default branch `main`, configured description and nine relevant topics. The reviewed implementation commit `f9da0772647ab7ec87c8f64e1324e06038b73973` matched the pushed commit. Required source files, reports, locks, docs, and all 19 actual screenshots were verified through the GitHub API. A credential/large-artifact/local-data scan of 97 tracked files found no flagged items.

[CI run 40130972332](https://github.com/sawantamar05/threadlens-nlp/actions/runs/40130972332) completed successfully, including the CPU-compatible test suite and offline synthetic demonstration rebuild/reload. Subsequent handoff documentation updates use the same workflow; the repository Actions page shows current checks.
