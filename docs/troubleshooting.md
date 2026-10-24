# Troubleshooting

| Symptom | Action |
|---|---|
| API unavailable in sidebar | Activate this project's environment and run `threadlens serve`. Check the printed API URL and `.local/logs/api.log`. Use `THREADLENS_API_URL` if starting UI separately. |
| Readiness 503 / no trained artifact | A clone contains no model weights. Run `threadlens demo` and `threadlens serve --demo` for the synthetic path, or reproduce collection/labeling/experiments for real models. A missing trained model is never replaced silently. |
| First prediction is slow | The API loads each selected artifact once. Initial transformer startup can be much slower than warm inference. Readiness loads the active model before launch completion. |
| Port already in use | Use `threadlens serve --api-port 8001 --ui-port 8502`. ThreadLens never kills another process to obtain a port. |
| MPS unavailable | Run `threadlens doctor` outside restricted execution environments; use `--device cpu` for training or `THREADLENS_DEVICE=cpu threadlens serve`. CPU operation is supported but slower. |
| GPU kernel failure | Inference reports CPU fallback. Training device requests are explicit; inspect the logged exception and repeat on CPU. `PYTORCH_ENABLE_MPS_FALLBACK=1` can enable PyTorch's unsupported-operation fallback before importing torch, but is not silently enabled. |
| Model download / TLS failure | Initial collection and Hugging Face downloads require normal internet access and trusted certificates. Use normal proxy/certificate settings; do not disable TLS verification. No paid inference key is required. |
| Offline use | Finish model download/training first. Saved local artifacts reload without network. Optionally set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Uncached HN threads still require internet. |
| HN invalid, missing, deleted, or inaccessible | Enter a positive story ID or `https://news.ycombinator.com/item?id=...`. Deleted/dead/empty comments are omitted. Retrieval visits a bounded pool and can return fewer rows. Public network failures return a readable error. |
| CSV rejected | Use UTF-8, include exactly one `text` column, non-empty text for every row, at most 1,000 rows / 2 MiB / 10,000 characters per comment. Export again from your editor if the encoding is wrong. |
| Weak-label classes missing in a split | Collect more suitable real source data under a new version. Do not invent labels or alter test references to force a target. |
| Frozen split differs | Do not overwrite the benchmark. Use a new `THREADLENS_ROOT` and dataset version for a separate experiment. |
| Improved experiment asks for a commit | Inspect `configs/improvement.json`, generated using train/validation evidence, then `git add configs/improvement.json && git commit -m 'Declare validation-selected improvement'`. Run `threadlens experiments --stage improved`. This makes the comparison decision auditable. |
| Independent label rejected | If an AI suggestion was shown, choose AI-assisted reviewed. Independent human provenance requires a person's initial blind label before suggestion exposure. Agent-driven review is AI-assisted. |
| Review corrections do not change benchmark | This is intentional. Revisions preserve original draft labels; create a separately versioned reviewed benchmark before retraining or claiming improved evaluation. |

Stop the foreground launcher with Ctrl+C, or run `.venv/bin/threadlens stop` from another terminal. For custom ports pass the same `--api-port` and `--ui-port`; use the same `THREADLENS_ROOT`. The command verifies the recorded process identity/start time before signaling it. Rerun the same `serve` command to restart. Inspect only ThreadLens's own logs; do not terminate unknown processes. Local texts and notes may be sensitive: keep `.local/`, `data/raw/`, and `artifacts/` untracked.
