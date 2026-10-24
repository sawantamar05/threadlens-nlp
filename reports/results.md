# Experiment report

These are measured agreements with unreviewed local NLI weak labels. Independent human semantic accuracy is **unmeasured**. The teacher is not evaluated against its own labels as an independent baseline.

Dataset: 720 comments / 36 threads. Frozen train/validation/test: 500/100/120 comments and 25/5/6 disjoint threads. Dataset/label hash: `b2b27c0cfe9b9b21248a53231e192aaad478879a347a7102b552d7f8f612a2e8`.

## Protocol

Quality goals were committed before test evaluation in `configs/preregistration.json`. Seeds 42, 7, and 2024 vary training randomness on fixed partitions; all seeds are reported. TF-IDF is fitted only on training text. Original DistilBERT: 3 epochs, learning rate 2e-5, batch 16, maximum 256 tokens, dynamic padding, AdamW, 0.01 weight decay, 10% linear warmup. Each run retains the epoch with highest validation macro F1; ties retain the earlier epoch. Only validation metrics select the final configuration. Seed 42 is the predeclared representative checkpoint, rather than a selected best seed.

Only seed 42 model artifacts are retained to conserve local disk; every seed keeps local row predictions, metadata, training curves, confusion counts and metrics. The model/tokenizer were saved, loaded with local-files-only, and checked for unchanged logits and fixed label mappings. MPS randomness is not guaranteed bitwise deterministic.

## Every measured run

| Configuration | Seed | Val macro F1 | Test accuracy | Test macro F1 | Request recall | Train s | Warm mean ms | Device |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| majority | 42 | 0.2698 | 0.6167 | 0.2543 | 0.0000 | 0.05 | 0.24 | cpu |
| majority | 7 | 0.2698 | 0.6167 | 0.2543 | 0.0000 | 0.03 | 0.24 | cpu |
| majority | 2024 | 0.2698 | 0.6167 | 0.2543 | 0.0000 | 0.03 | 0.24 | cpu |
| tfidf | 42 | 0.2675 | 0.6500 | 0.3346 | 0.0000 | 0.05 | 0.31 | cpu |
| tfidf | 7 | 0.2675 | 0.6500 | 0.3346 | 0.0000 | 0.05 | 0.32 | cpu |
| tfidf | 2024 | 0.2675 | 0.6500 | 0.3346 | 0.0000 | 0.05 | 0.32 | cpu |
| distilbert-original | 42 | 0.3175 | 0.6250 | 0.2736 | 0.0000 | 78.29 | 11.10 | mps |
| distilbert-original | 7 | 0.3175 | 0.6417 | 0.3073 | 0.0000 | 93.59 | 12.48 | mps |
| distilbert-original | 2024 | 0.2698 | 0.6167 | 0.2543 | 0.0000 | 102.60 | 18.89 | mps |
| distilbert-improved | 42 | 0.6047 | 0.6167 | 0.5003 | 0.5000 | 83.54 | 12.03 | mps |
| distilbert-improved | 7 | 0.4543 | 0.6583 | 0.3804 | 0.0000 | 130.79 | 21.10 | mps |
| distilbert-improved | 2024 | 0.6603 | 0.6500 | 0.5842 | 0.5000 | 128.44 | 10.74 | mps |

## Variation across seeds

Population standard deviation describes these three runs, not a confidence interval or uncertainty from resampling discussions.

| Configuration | Test accuracy mean ± std | Test macro F1 mean ± std | Mean validation macro F1 |
|---|---:|---:|---:|
| majority | 0.6167 ± 0.0000 | 0.2543 ± 0.0000 | 0.2698 |
| tfidf | 0.6500 ± 0.0000 | 0.3346 ± 0.0000 | 0.2675 |
| distilbert-original | 0.6278 ± 0.0104 | 0.2784 ± 0.0219 | 0.3016 |
| distilbert-improved | 0.6417 ± 0.0180 | 0.4883 ± 0.0836 | 0.5731 |

## One bounded improvement

Declared `2026-10-24T20:13:47.956570+00:00` in `configs/improvement.json`: **class-weighted cross-entropy**. Validation mean recall for request was 0.0000; train-only inverse-frequency weights address neglected classes.

Before: `{"batch_size": 16, "class_weighted_loss": false, "epochs": 3, "learning_rate": 2e-05, "max_length": 256}`.

After: `{"batch_size": 16, "class_weighted_loss": true, "epochs": 3, "learning_rate": 2e-05, "max_length": 256}`.

The declaration was committed before improved test evaluation. No further search was performed. Both configurations use exactly the same test comments. An improvement hypothesis is retained even when measured performance does not improve.

## Per-label metrics and confusion counts

Representative seed 42 runs. Confusion rows are reference labels; columns are predictions in analysis/request/commentary order. Full per-seed arrays are in `reports/experiments.json`.

### majority

| Label | Precision | Recall | F1 | Support | Predicted count |
|---|---:|---:|---:|---:|---:|
| analysis | 0.0000 | 0.0000 | 0.0000 | 38 | 0 |
| request | 0.0000 | 0.0000 | 0.0000 | 8 | 0 |
| commentary | 0.6167 | 1.0000 | 0.7629 | 74 | 120 |

```text
[0, 0, 38]
[0, 0, 8]
[0, 0, 74]
```

### tfidf

| Label | Precision | Recall | F1 | Support | Predicted count |
|---|---:|---:|---:|---:|---:|
| analysis | 0.8333 | 0.1316 | 0.2273 | 38 | 6 |
| request | 0.0000 | 0.0000 | 0.0000 | 8 | 0 |
| commentary | 0.6404 | 0.9865 | 0.7766 | 74 | 114 |

```text
[5, 0, 33]
[0, 0, 8]
[1, 0, 73]
```

### distilbert-original

| Label | Precision | Recall | F1 | Support | Predicted count |
|---|---:|---:|---:|---:|---:|
| analysis | 0.5000 | 0.0263 | 0.0500 | 38 | 2 |
| request | 0.0000 | 0.0000 | 0.0000 | 8 | 0 |
| commentary | 0.6271 | 1.0000 | 0.7708 | 74 | 118 |

```text
[1, 0, 37]
[1, 0, 7]
[0, 0, 74]
```

### distilbert-improved

| Label | Precision | Recall | F1 | Support | Predicted count |
|---|---:|---:|---:|---:|---:|
| analysis | 0.5200 | 0.3421 | 0.4127 | 38 | 25 |
| request | 0.2667 | 0.5000 | 0.3478 | 8 | 15 |
| commentary | 0.7125 | 0.7703 | 0.7403 | 74 | 80 |

```text
[13, 5, 20]
[1, 4, 3]
[11, 6, 57]
```

## Selection, goals, and error analysis

Selected local transformer: `distilbert-improved-seed42`, by mean validation macro F1. Most frequent seed 42 test disagreements: analysis → commentary: 20; commentary → analysis: 11; commentary → request: 6.

These are disagreements with silver references, not confirmed semantic errors. No public raw-text examples are redistributed. Local `.local/predictions/*-test.jsonl` preserves IDs, full source text, reference/predicted labels and scores for review. Inspect mixed-purpose comments, rhetorical questions and unsupported numerical claims in blind review before drawing semantic conclusions.

Input truncation in selected run: `{"train": 18, "validation": 5, "test": 3}`. Complete source text remains stored. A request or supporting evidence after token 256 can be missed by the trained model, even though the windowed teacher considered the full comment.

Only 4 validation and 8 test records are weak-labeled request; minority metrics are unstable. Class imbalance favors commentary. Accuracy must be considered alongside macro F1, per-label recall and prediction counts.

| Preregistered goal | Target | Observed | Met |
|---|---:|---:|---|
| silver_macro_f1 | 0.65 | 0.4883 | False |
| silver_request_recall | 0.5 | 0.5000 | True |
| warm_single_latency_ms | 500 | 12.0265 | True |

Semantic quality remains unmeasured even when a silver-agreement or latency goal passes. Functional software checks are separate from these model-quality goals.

## Latency and reproducibility

Warm latency uses up to 20 sequential single-comment requests after 3 warmups, including tokenization and model inference, excluding HTTP/UI overhead. MPS is synchronized around each timer. Models use dynamic input lengths up to 256 tokens. scikit-learn runs use CPU. Original project inspection snapshot (future run hardware may differ): `{"architecture": "arm64", "cpu_cores": 8, "disk_available_gib_at_start": 19, "hardware": "Apple M3 MacBook Air", "inspection_date": "2026-10-24", "memory_gib": 16, "mps_verified_available": true, "os": "macOS 26.6.2", "python": "3.11.0", "scope": "Original recorded local experiments; future hardware may differ"}`. Complete package versions, exact base-model revision, artifact identifier, dataset hash, split reference, source commit, per-epoch losses and durations are embedded in each run's metadata.

Run `threadlens collect`, `threadlens label --device mps`, `threadlens split`, then `threadlens experiments --stage initial --device mps`. Commit the generated `configs/improvement.json` before `threadlens experiments --stage improved --device mps`. Use `--device cpu` on other platforms. See the README for locked setup and the data card for source-edit limitations. `reports/per_seed_metrics.csv` and `reports/experiments.json` are shareable aggregate artifacts.
