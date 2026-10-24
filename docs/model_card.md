# Model card

ThreadLens is an original local discourse-classification portfolio application by Amar Sawant, implemented with AI assistance. It predicts analysis/request/commentary; it does not verify factual truth, rate authors, or establish content quality.

## Selected artifact

- Identifier: `distilbert-improved-seed42-b2b27c0cfe9b`; selected configuration: `distilbert-improved`; fixed representative seed: 42.
- Base: [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased), Apache-2.0, revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.
- Label mapping: analysis=0, request=1, commentary=2.
- Actual training device: `mps`; environment: `{"architecture": "arm64", "cpu_cores": 8, "disk_available_gib_at_start": 19, "hardware": "Apple M3 MacBook Air", "inspection_date": "2026-10-24", "memory_gib": 16, "mps_verified_available": true, "os": "macOS 26.6.2", "python": "3.11.0", "scope": "Original recorded local experiments; future hardware may differ"}`; CPU supported through the same training and inference code. Auto mode chooses MPS when available, otherwise CPU; model metadata and doctor output expose the actual choice. Explicit MPS mode fails clearly if unavailable.
- Source commit: `e8cbc23bbb6b58a099225ef3af86c2b7a9a756a2`. Dataset/label hash: `b2b27c0cfe9b9b21248a53231e192aaad478879a347a7102b552d7f8f612a2e8`.
- Training configuration: `{"base_license": "apache-2.0", "base_model": "distilbert/distilbert-base-uncased", "base_revision": "12040accade4e8a0f71eabdb258fecc2e7e948be", "batch_size": 16, "checkpoint_selection": "highest validation macro F1, earliest epoch wins ties", "class_weighted_loss": true, "dynamic_padding": true, "epochs": 3, "gradient_clip_norm": 1.0, "learning_rate": 2e-05, "max_length": 256, "randomness": "Training seeds vary initialization, dropout, shuffle on fixed thread partitions; MPS kernels are not guaranteed bitwise deterministic.", "seed": 42, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
- Duration: 83.54 seconds; chosen epoch: 3. Package versions: `{"huggingface-hub": "0.36.2", "numpy": "2.4.6", "pandas": "2.3.3", "scikit-learn": "1.9.1", "tokenizers": "0.22.2", "torch": "2.14.1", "transformers": "4.57.6"}`.

Weights, tokenizer, local database and raw comments are excluded from normal Git. Run the documented collection/labeling/splitting/training commands to rebuild. The artifact was saved with safetensors, reloaded locally, and checked for fixed label mapping and matching logits. The API caches each requested model artifact once per process; Streamlit calls the API and does not load a second copy.

## Measured agreement, not independent accuracy

All 720 source labels are unreviewed `weak_label` outputs from a separate local NLI model; there are zero independent human benchmark labels. The selected seed 42 test agreement is accuracy=0.6167, macro F1=0.5003 on 120 comments from 6 held-out threads. Its warm mean single-comment model latency was 12.03ms over 20 requests after 3 warmups, excluding network/UI overhead. See [all seeds and the bounded before/after experiment](experiments.md); seed 42 is not a best-seed selection.

Configuration selection uses mean validation macro F1 across seeds 42, 7, and 2024. Test data is held out of tuning. Same-thread comments never span partitions. Seeds measure optimization randomness on fixed groups and do not estimate split uncertainty. The test has only 8 weak-labeled requests; minority-class estimates are unstable. The teacher is never treated as an independent baseline against labels it generated.

## Scores, truncation, and limitations

Scores are uncalibrated softmax outputs, not probabilities that a prediction is correct. A fixed 0.60 maximum-score threshold flags low confidence; it was not tuned on test data. Neural explanations are not invented. Majority and TF-IDF alternatives use their own documented score outputs.

Complete comments are stored locally. DistilBERT sees at most 256 tokens, and inference responses/UI explicitly flag truncation. Selected training/validation/test truncation counts: `{"train": 18, "validation": 5, "test": 3}`. A late request or supporting evidence may be missed. Mixed-purpose comments, rhetorical questions, quotation attribution, unsupported numbers, sarcasm, code snippets and context-dependent replies are difficult. The model sees a comment alone rather than its parent/reply context. Pretraining and the English-only HN collection introduce demographic, topical and linguistic biases. No independent human validation, production deployment, real users or business impact has been established.

The demo classifier is trained only on original synthetic fixtures and has `synthetic_demo` provenance. It is an offline software demonstration and is separate from this real-source weak-label benchmark. The annotation UI supports independent blind first labels and records revisions, enabling a future explicitly versioned human benchmark without relabeling this frozen experiment after inspecting errors.

Intended use is exploratory discussion analysis with human review. Do not use these labels as automated moderation decisions, factual verification, employee assessment, or evidence of a person's expertise. See [the data card](data_card.md) for source retention and taxonomy ambiguity and the API's `/models` response for active runtime metadata.
