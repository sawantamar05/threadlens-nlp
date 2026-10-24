"""ThreadLens browser application. All inference and review state live behind the API."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

from threadlens.config import DEFINITIONS, LABELS, settings
from threadlens.data.csv_io import safe_csv, validate_csv

st.set_page_config(
    page_title="ThreadLens · Discussion Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("THREADLENS_API_URL", "http://127.0.0.1:8000").rstrip("/")
COLORS = {"analysis": "#16766F", "request": "#3A67A5", "commentary": "#8C6D3F"}
PROVENANCE = {
    "weak_label": "Automatically generated · weak labels",
    "ai_assisted_reviewed": "AI-assisted review",
    "human_independent": "Independent human labels",
    "synthetic_demo": "Synthetic demonstration",
}
PAGES = (
    "Classify a comment",
    "Analyze a thread",
    "Batch classification",
    "Experiments",
    "Annotation review",
    "About & models",
)


class APIError(RuntimeError):
    """A user-facing local service error."""


def api(method: str, path: str, **kwargs: Any) -> dict:
    """Only the local API owns models and annotation persistence."""
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=180, **kwargs)
    except httpx.RequestError as exc:
        raise APIError(
            "The local API is unavailable. Start both services with `threadlens serve` "
            "from the project environment, then select Refresh models. "
            f"Connection: {API_URL}."
        ) from exc
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text
        if isinstance(detail, list):
            detail = "; ".join(
                f"{'.'.join(map(str, entry.get('loc', [])))}: {entry.get('msg', '')}"
                for entry in detail
            )
        if isinstance(detail, dict):
            detail = json.dumps(detail)
        raise APIError(f"{detail} (HTTP {response.status_code})")
    try:
        return response.json()
    except ValueError as exc:
        raise APIError("The API returned an unreadable response. Check the API log.") from exc


def notices(warnings: Iterable[str]) -> None:
    for warning in dict.fromkeys(warnings):
        st.warning(warning)


def taxonomy() -> None:
    st.markdown("**Read the contribution, then apply this order:**")
    st.markdown(
        "1. A genuine primary request → **request**\n"
        "2. Otherwise, a claim with relevant support → **analysis**\n"
        "3. Otherwise → **commentary**"
    )
    for label in ("request", "analysis", "commentary"):
        st.markdown(f"**{label.capitalize()}** — {DEFINITIONS[label]}")
    st.caption(
        "Rhetorical questions are not automatically requests. A number, a link, technical "
        "vocabulary, or length alone does not establish analysis. This categorizes discourse; "
        "it does not check whether a claim is true."
    )


def score_chart(scores: dict[str, float]) -> None:
    frame = pd.DataFrame({"Label": LABELS, "Score": [scores.get(label, 0) for label in LABELS]})
    fig = px.bar(
        frame,
        x="Score",
        y="Label",
        orientation="h",
        color="Label",
        color_discrete_map=COLORS,
        text="Score",
        range_x=[0, 1],
    )
    fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=230,
        showlegend=False,
        margin=dict(l=0, r=45, t=10, b=0),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(LABELS))),
        xaxis=dict(title="Model score · uncalibrated", tickformat=".0%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def result_card(result: dict) -> None:
    with st.container(border=True):
        label_col, time_col, input_col = st.columns([2, 1, 1])
        label_col.metric("Predicted contribution", result["label"].capitalize())
        time_col.metric("Processing time", f"{result['processing_ms']:.0f} ms")
        input_col.metric("Input", "Truncated" if result["truncated"] else "Complete")
        st.caption(f"Model: {result['model_id']} · version {result['model_version']}")
        if result.get("low_confidence"):
            st.warning("Low confidence: the highest score is below 0.60. Review this prediction.")
        if result.get("truncated"):
            st.warning(
                "The model used only the beginning of this comment. A request or supporting "
                "evidence near the end may have been missed. The entered text is preserved."
            )
        score_chart(result["scores"])
        notices(result.get("warnings", []))
        st.caption(
            "Scores express the model's relative preference among these labels. They are "
            "not calibrated probabilities that the classification is correct."
        )


def require_model(model_id: str | None) -> bool:
    if not model_id:
        st.info(
            "No trained artifact is available. Run `threadlens demo` for a clearly labeled "
            "offline demonstration, or rebuild the benchmark artifacts using the README."
        )
        return False
    return True


def classify(model_id: str | None) -> None:
    st.header("Classify a comment")
    st.write("See how a complete English comment contributes to a technical discussion.")
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        with st.form("classify_form"):
            text = st.text_area(
                "Complete comment",
                height=235,
                max_chars=settings.max_chars,
                placeholder="Paste the full comment, including the context and supporting details…",
                help="Up to 10,000 characters. Transformer token limits are reported in the result.",
            )
            submitted = st.form_submit_button(
                "Classify comment", type="primary", disabled=not model_id, width="stretch"
            )
        if submitted:
            st.session_state.pop("single_result", None)
            if not text.strip():
                st.error("Enter a non-empty comment before classifying.")
            else:
                with st.spinner("Reading the contribution…"):
                    try:
                        st.session_state.single_result = api(
                            "POST", "/predict", json={"text": text, "model_id": model_id}
                        )
                    except APIError as exc:
                        st.error(str(exc))
        if st.session_state.get("single_result"):
            result_card(st.session_state.single_result)
        require_model(model_id)
    with right:
        with st.container(border=True):
            st.subheader("Three kinds of contribution")
            taxonomy()


def thread_page(model_id: str | None) -> None:
    st.header("Analyze a Hacker News thread")
    st.write("Explore the mix of contributions in a bounded sample from one discussion.")
    with st.form("thread_form"):
        story = st.text_input(
            "Story ID or Hacker News URL",
            placeholder="https://news.ycombinator.com/item?id=…",
            help="Only Hacker News story IDs and news.ycombinator.com item URLs are supported.",
        )
        limit = st.slider("Maximum comments", min_value=1, max_value=100, value=20, step=1)
        submitted = st.form_submit_button("Analyze thread", type="primary", disabled=not model_id)
    if submitted:
        st.session_state.pop("thread_result", None)
        if not story.strip():
            st.error("Enter a Hacker News story ID or story URL.")
        else:
            with st.spinner("Retrieving public comments and classifying the sample…"):
                try:
                    st.session_state.thread_result = api(
                        "POST",
                        "/analyze-thread",
                        json={"story": story, "limit": limit, "model_id": model_id},
                    )
                except APIError as exc:
                    st.error(str(exc))
    result = st.session_state.get("thread_result")
    if not result:
        require_model(model_id)
        return
    notices(result.get("warnings", []))
    st.subheader(result.get("title") or f"Hacker News story {result['story_id']}")
    st.link_button(
        "Open source discussion", f"https://news.ycombinator.com/item?id={result['story_id']}"
    )
    comments = result.get("comments", [])
    if not comments:
        st.info(
            "This story has no usable comments in the bounded retrieval. Deleted, dead, and empty comments are excluded."
        )
        return
    summary = st.columns(3)
    for column, label in zip(summary, LABELS, strict=True):
        count = result.get("counts", {}).get(label, 0)
        proportion = result.get("proportions", {}).get(label, count / len(comments))
        column.metric(
            label.capitalize(),
            f"{count}",
            f"{proportion:.1%} of retrieved comments",
            delta_color="off",
        )
    st.caption(
        f"{len(comments)} retrieved comments · predictions from {model_id}. "
        "This bounded sample may not represent the full thread. Predictions are not reviewed reference labels."
    )
    selected = st.multiselect("Filter predicted labels", options=list(LABELS), default=list(LABELS))
    visible = [comment for comment in comments if comment["label"] in selected]
    st.caption(f"Showing {len(visible)} of {len(comments)} comments")
    for comment in visible:
        preview = " ".join(comment["text"].split())[:110]
        with st.expander(f"{comment['label'].capitalize()} · {preview}"):
            st.write(comment["text"])
            if comment.get("permalink"):
                st.link_button("Open source comment", comment["permalink"])
            st.caption(
                f"Prediction: {comment['label']} · model {comment.get('model_version', model_id)}"
            )
            if comment.get("reviewed_label"):
                st.caption(
                    f"Reviewed reference: {comment['reviewed_label']} · "
                    f"provenance {comment.get('reviewed_provenance', 'unspecified')}"
                )
            if comment.get("truncated"):
                st.warning(
                    "Input was truncated for this model; complete source text is shown above."
                )
            score_chart(comment["scores"])
            notices(comment.get("warnings", []))


def append_predictions(frame: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    """Keep rows and source columns intact, including names that resemble output columns."""
    if len(frame) != len(results):
        raise ValueError("The API returned a different number of predictions than input rows.")
    output = frame.copy()
    fields = ["label", "model_id", "model_version", "truncated", "processing_ms", "low_confidence"]
    values = {field: [result.get(field) for result in results] for field in fields}
    for label in LABELS:
        values[f"score_{label}"] = [result["scores"][label] for result in results]
    values["warnings"] = ["; ".join(result.get("warnings", [])) for result in results]
    for field, value in values.items():
        name = f"threadlens_{field}"
        while name in output.columns:
            name = f"_{name}"
        output[name] = value
    return output


def batch_page(model_id: str | None) -> None:
    st.header("Batch classification")
    st.write(
        "Upload a CSV with a `text` column. Results retain the original row order and columns."
    )
    st.caption(
        "Limits: 2 MB per file · 1,000 rows · 10,000 characters per comment · 100 comments per API batch."
    )
    st.download_button(
        "Download synthetic sample CSV",
        'text\n"Could you explain which configuration controls retries?"\n'
        '"Caching reduced our repeated query latency from 80 ms to 12 ms, because the result bypasses storage."\n'
        '"I really like this interface."\n',
        file_name="threadlens_synthetic_sample.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload comments CSV", type=["csv"])
    frame = None
    fingerprint = None
    if uploaded is not None:
        try:
            content = uploaded.getvalue()
            frame = validate_csv(content)
            fingerprint = hashlib.sha256(content + str(model_id).encode()).hexdigest()
            st.success(
                f"Validated {len(frame)} rows. All comments are non-empty and within the input limit."
            )
            st.dataframe(frame.head(8), hide_index=True, width="stretch")
        except ValueError as exc:
            st.error(str(exc))
    if fingerprint != st.session_state.get("batch_fingerprint"):
        st.session_state.pop("batch_result", None)
    run = st.button("Classify CSV", type="primary", disabled=frame is None or not model_id)
    if run and frame is not None:
        st.session_state.pop("batch_result", None)
        progress = st.progress(0.0, text="Preparing batches…")
        predictions = []
        try:
            for start in range(0, len(frame), settings.max_batch):
                chunk = frame["text"].iloc[start : start + settings.max_batch].tolist()
                response = api(
                    "POST", "/predict/batch", json={"texts": chunk, "model_id": model_id}
                )
                predictions.extend(response["results"])
                progress.progress(
                    len(predictions) / len(frame),
                    text=f"Classified {len(predictions)} of {len(frame)} comments",
                )
            st.session_state.batch_result = append_predictions(frame, predictions)
            st.session_state.batch_fingerprint = fingerprint
            st.session_state.batch_predictions = predictions
        except (APIError, ValueError) as exc:
            st.error(
                f"Batch stopped after {len(predictions)} rows: {exc}. Retry after resolving the issue."
            )
        finally:
            progress.empty()
    result = st.session_state.get("batch_result")
    if result is not None:
        st.success(f"Completed {len(result)} comments in original row order.")
        counts = pd.Series(
            [row["label"] for row in st.session_state.batch_predictions]
        ).value_counts()
        for column, label in zip(st.columns(3), LABELS, strict=True):
            column.metric(label.capitalize(), int(counts.get(label, 0)))
        st.dataframe(result, hide_index=True, width="stretch")
        st.download_button(
            "Download predictions CSV",
            safe_csv(result),
            file_name="threadlens_predictions.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption(
            "Spreadsheet formula-like cells are escaped in the export. Scores are uncalibrated. Input columns are preserved; output fields use a threadlens_ prefix."
        )
    require_model(model_id)


def metric_value(metrics: dict, key: str) -> float:
    return float(metrics.get(key, 0))


def experiment_page() -> None:
    st.header("Experiments")
    st.write("Compare measured runs on fixed discussion-level splits.")
    try:
        report = api("GET", "/experiments")
    except APIError as exc:
        st.info(str(exc))
        return
    dataset = report.get("dataset", {})
    provenance = dataset.get("provenance", report.get("provenance", "weak_label"))
    if provenance != "human_independent":
        st.warning(
            f"Benchmark provenance: {PROVENANCE.get(provenance, provenance)}. "
            "These metrics measure agreement with reference labels, not independently validated semantic accuracy."
        )
    else:
        st.info(
            "Benchmark provenance: independent human annotations. See the report for review scope and limitations."
        )
    st.caption(f"Model-quality status: {report.get('quality_status', 'See report')}")
    counts = dataset.get("label_counts", dataset.get("labels", {}))
    total = dataset.get(
        "comment_count",
        dataset.get(
            "comments", dataset.get("total_comments", sum(counts.values()) if counts else 0)
        ),
    )
    threads = dataset.get("thread_count", dataset.get("threads", 0))
    cols = st.columns(4)
    cols[0].metric("Comments", total)
    cols[1].metric("Discussion threads", threads)
    cols[2].metric("Labels", len(LABELS))
    cols[3].metric("Training seeds", "42 · 7 · 2024")
    if counts:
        label_frame = pd.DataFrame({"Label": list(counts), "Count": list(counts.values())})
        fig = px.bar(label_frame, x="Label", y="Count", color="Label", color_discrete_map=COLORS)
        fig.update_layout(showlegend=False, height=230, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "All training seeds share the same frozen thread groups. Seeds measure training randomness; they do not create independent test datasets."
    )
    runs = report.get("runs", [])
    if not runs:
        st.info("No completed experiment runs are recorded yet.")
    else:
        rows = []
        for run in runs:
            test = run.get("test", run.get("test_metrics", {}))
            validation = run.get("validation", run.get("validation_metrics", {}))
            rows.append(
                {
                    "Model": run.get("model_id", run.get("model", "unknown")),
                    "Configuration": run.get(
                        "configuration", run.get("config_name", run.get("config", "original"))
                    ),
                    "Seed": run.get("seed", "—"),
                    "Validation macro F1": metric_value(validation, "macro_f1"),
                    "Test accuracy": metric_value(test, "accuracy"),
                    "Test macro F1": metric_value(test, "macro_f1"),
                    "Warm inference (ms)": run.get("latency", {}).get("warm_single_mean_ms"),
                    "Device": run.get("latency", {}).get("device", "—"),
                }
            )
        comparison = pd.DataFrame(rows)
        if comparison["Configuration"].map(lambda value: isinstance(value, dict)).any():
            comparison["Configuration"] = comparison["Configuration"].map(
                lambda value: (
                    json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                )
            )
        st.subheader("All runs")
        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="%.3f")
                for name in ("Validation macro F1", "Test accuracy", "Test macro F1")
            },
        )
        chart_frame = comparison.copy()
        chart_frame["Run"] = (
            chart_frame["Model"].astype(str) + " · seed " + chart_frame["Seed"].astype(str)
        )
        fig = px.bar(chart_frame, x="Run", y="Test macro F1", color="Configuration", range_y=[0, 1])
        fig.update_layout(
            height=340, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="Configuration"
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        run_index = st.selectbox(
            "Inspect a run",
            range(len(runs)),
            format_func=lambda i: (
                f"{rows[i]['Model']} · {rows[i]['Configuration']} · seed {rows[i]['Seed']}"
            ),
        )
        selected = runs[run_index]
        test = selected.get("test", selected.get("test_metrics", {}))
        left, right = st.columns(2)
        per_label = test.get("per_label", test.get("classification_report", {}))
        with left:
            st.subheader("Test results by label")
            if per_label:
                if isinstance(per_label, list):
                    st.dataframe(pd.DataFrame(per_label), hide_index=True, width="stretch")
                else:
                    label_rows = [
                        {"Label": label, **per_label[label]}
                        for label in LABELS
                        if label in per_label
                    ]
                    st.dataframe(pd.DataFrame(label_rows), hide_index=True, width="stretch")
        with right:
            confusion = test.get("confusion_matrix", test.get("confusion", []))
            st.subheader("Confusion counts")
            if confusion:
                fig = px.imshow(
                    confusion,
                    x=list(LABELS),
                    y=list(LABELS),
                    text_auto=True,
                    color_continuous_scale="Teal",
                    labels={"x": "Predicted label", "y": "Reference label", "color": "Count"},
                )
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with st.expander("Run configuration, timing, and provenance"):
            st.json(selected)
    summaries = report.get("summaries", [])
    if isinstance(summaries, list) and summaries:
        st.subheader("Across training seeds")
        st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch")
        st.caption(
            "Variation is measured across the recorded seeds on the same frozen test examples."
        )
    st.subheader("Declared improvement")
    improvement = report.get("improvement", {})
    if isinstance(improvement, dict):
        if improvement.get("reason") or improvement.get("rationale"):
            st.write(improvement.get("reason", improvement.get("rationale")))
        st.json(improvement)
    else:
        st.write(improvement)
    with st.expander("Summary variation and split manifest"):
        st.json({"summaries": report.get("summaries", {}), "splits": report.get("splits", {})})
    st.download_button(
        "Download experiment report",
        json.dumps(report, indent=2),
        file_name="threadlens_experiments.json",
        mime="application/json",
    )


def review_page() -> None:
    st.header("Annotation review")
    st.write("Record an initial label or a later correction while preserving the original draft.")
    st.info(
        "Reviewing a model suggestion produces AI-assisted provenance. Only a person labeling independently before seeing suggestions can create an independent human annotation."
    )
    blind = st.toggle(
        "Blind labeling mode",
        value=True,
        help="Hide draft labels and scores until an initial label is saved.",
    )
    if st.session_state.get("review_blind") != blind:
        st.session_state.pop("review_item", None)
        st.session_state.pop("review_history", None)
        st.session_state.review_blind = blind
    if st.button("Load next comment", type="primary"):
        st.session_state.pop("review_item", None)
        st.session_state.pop("review_history", None)
    if "review_item" not in st.session_state:
        try:
            response = api("GET", "/review/next", params={"blind": str(blind).lower()})
            st.session_state.review_item = response.get("item")
        except APIError as exc:
            st.error(str(exc))
            return
    item = st.session_state.get("review_item")
    if not item:
        st.info(
            "No unreviewed comments are available. Collect and weak-label fresh data, then prepare the review queue with `threadlens prepare-review`."
        )
        return
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.caption(f"Comment {item['comment_id']} · thread {item.get('thread_id', '—')}")
        with st.container(border=True):
            st.write(item["text"])
        if item.get("permalink"):
            st.link_button("Read source context", item["permalink"])
        if item.get("draft_label") and not blind:
            st.caption(
                f"Original draft: {item['draft_label']} · {item.get('draft_provenance', 'weak_label')}"
            )
            if item.get("draft_scores"):
                score_chart(item["draft_scores"])
        elif blind and not st.session_state.get("review_history"):
            st.caption("Draft label and scores are hidden until your initial submission.")
        choices = ["AI-assisted reviewed"]
        if (
            blind
            and not item.get("latest_annotation")
            and not st.session_state.get("review_history")
        ):
            choices.append("Human independent")
        provenance = st.selectbox(
            "Annotation provenance",
            choices,
            help="Use Human independent only for a person's initial label assigned without seeing an AI suggestion.",
        )
        independent = False
        if provenance == "Human independent":
            independent = st.checkbox(
                "I am a human reviewer and assigned this initial label without seeing any model suggestion."
            )
        with st.form(f"review_form_{item['comment_id']}"):
            label = st.radio("Assign a label", list(LABELS), index=None, horizontal=True)
            notes = st.text_area("Review notes (optional)", max_chars=2000, height=90)
            saved = st.form_submit_button("Save annotation", type="primary")
        if saved:
            if label is None:
                st.error("Choose a label before saving.")
            elif provenance == "Human independent" and not independent:
                st.error("Confirm independent human review, or select AI-assisted reviewed.")
            else:
                try:
                    annotation = api(
                        "POST",
                        "/annotations",
                        json={
                            "comment_id": str(item["comment_id"]),
                            "label": label,
                            "provenance": "human_independent"
                            if provenance == "Human independent"
                            else "ai_assisted_reviewed",
                            "blind": blind,
                            "notes": notes,
                        },
                    )
                    item["latest_annotation"] = annotation
                    st.session_state.review_history = api(
                        "GET",
                        f"/annotations/{item['comment_id']}",
                        params={"blind": str(blind).lower()},
                    )
                    st.success(
                        f"Saved revision {annotation['revision']} · {annotation['provenance']}. Original draft preserved."
                    )
                except APIError as exc:
                    st.error(str(exc))
        if st.button("Reload saved revisions"):
            try:
                st.session_state.review_history = api(
                    "GET",
                    f"/annotations/{item['comment_id']}",
                    params={"blind": str(blind).lower()},
                )
            except APIError as exc:
                st.error(str(exc))
        history = st.session_state.get("review_history")
        if history:
            st.subheader("Saved revision history")
            if history.get("draft"):
                with st.expander("Original immutable draft"):
                    st.json(history["draft"])
            revisions = history.get("revisions", [])
            if revisions:
                st.dataframe(pd.DataFrame(revisions), hide_index=True, width="stretch")
            else:
                st.caption("No review revisions have been saved for this comment.")
    with right:
        with st.container(border=True):
            taxonomy()
        st.caption(
            "Revisions append to the local SQLite database. They do not silently modify the frozen benchmark or any published results."
        )


def about_page(models: list[dict], selected: dict | None) -> None:
    st.header("About ThreadLens")
    st.write(
        "A local discussion intelligence application by Amar Sawant. ThreadLens makes the kinds of contributions in a technical conversation easier to inspect."
    )
    st.subheader("One inference service, clear boundaries")
    st.markdown(
        "**Streamlit interface → FastAPI → shared application services → local model artifacts**\n\n"
        "Hacker News API retrieval and local SQLite state support data collection and review. "
        "PyTorch and Hugging Face Transformers train DistilBERT; scikit-learn supplies a "
        "majority baseline and TF-IDF with logistic regression. The interface calls the API "
        "and keeps no second copy of the transformer. No paid inference account is required."
    )
    st.subheader("Active model information")
    if selected:
        st.caption(
            f"{selected['id']} · {selected.get('version', 'version unspecified')} · {PROVENANCE.get(selected.get('provenance'), selected.get('provenance', 'unspecified'))}"
        )
        st.json(selected)
    else:
        st.info("No model is selected. Build local artifacts or run the synthetic demonstration.")
    with st.expander("All registered models"):
        st.json(models)
    st.subheader("Intended use and limits")
    st.markdown(
        "Use the labels to browse and summarize English technical discussion. Mixed-purpose "
        "comments, rhetorical questions, sarcasm, missing parent context, and support appearing "
        "late in a comment can be ambiguous. Truncated transformer input can omit that context. "
        "Scores are uncalibrated; labels do not establish factual truth, author expertise, "
        "or the quality of a person.\n\n"
        "The current real-data benchmark uses automatically generated weak labels unless a "
        "report explicitly says otherwise. Agreement with that benchmark is provisional. "
        "Synthetic demonstration data is separate from real-data evaluation. Independent "
        "human validation remains necessary before making semantic accuracy claims."
    )
    st.subheader("Data and attribution")
    st.markdown(
        "Comments come from the [official Hacker News API](https://github.com/HackerNews/API). "
        "Complete source text, local databases, caches, and model weights remain local. "
        "Original code and synthetic examples are distributed under MIT; pretrained models "
        "retain their own licenses. Source content may later be edited or deleted. "
        "See the repository data card, model card, experiment protocol, and third-party notices.\n\n"
        "Built with AI assistance. Automated annotation and assistant-driven testing are "
        "disclosed and are never treated as independent human labeling."
    )
    with st.expander("Label guide"):
        taxonomy()
    st.link_button("Open API documentation", f"{API_URL}/docs")


def main() -> None:
    st.markdown(
        """<style>
        .stApp {background:#F6F8FA; color:#182B43;}
        [data-testid="stSidebar"] {background:#EAF0F4;}
        [data-testid="stSidebar"] h1 {color:#142A43; letter-spacing:-1px;}
        .block-container {max-width:1240px; padding-top:2.2rem; padding-bottom:3rem;}
        h1,h2,h3 {color:#142A43; letter-spacing:-.025em;}
        [data-testid="stMetricValue"] {font-size:1.7rem;}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, [data-testid="stCaption"], [data-testid="stCaption"] p, .stCaption, .stCaption p {color:#52647A !important;}
        [data-testid="stCaptionContainer"], [data-testid="stCaption"], .stCaption {opacity:1 !important;}
        div[data-testid="stForm"] {background:#FFFFFF;}
        div[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:12px;}
        .tl-kicker {color:#16766F; font-size:.76rem; font-weight:700; letter-spacing:.14em; margin:0 0 .25rem;}
        .tl-subtitle {color:#526174; margin:0 0 1.5rem;}
        @media (max-width:700px){.block-container {padding:1.3rem 1rem 2rem;}[data-testid="stMetricValue"] {font-size:1.35rem;}}
        </style>""",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.title("ThreadLens")
        st.caption("NLP DISCUSSION INTELLIGENCE")
        st.divider()
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        st.subheader("Inference model")
        if st.button("Refresh models", width="stretch"):
            st.session_state.pop("models_error", None)
        try:
            registry = api("GET", "/models")
            models = registry.get("models", [])
            st.session_state.pop("models_error", None)
        except APIError as exc:
            models = []
            registry = {}
            st.session_state.models_error = str(exc)
        available = [model for model in models if model.get("available", True)]
        model_id = None
        selected = None
        if available:
            ids = [model["id"] for model in available]
            active = registry.get("active_model")
            default = ids.index(active) if active in ids else 0
            model_id = st.selectbox(
                "Select model",
                ids,
                index=default,
                format_func=lambda value: next(
                    (
                        model.get("name") or model["id"]
                        for model in available
                        if model["id"] == value
                    ),
                    value,
                ),
            )
            selected = next(model for model in available if model["id"] == model_id)
            st.caption(f"Version {selected.get('version', 'unspecified')}")
            st.caption(
                PROVENANCE.get(
                    selected.get("provenance"), selected.get("provenance", "Provenance unspecified")
                )
            )
        elif st.session_state.get("models_error"):
            st.error(st.session_state.models_error)
        else:
            st.warning("No trained artifacts found. Run `threadlens demo` or train a model.")
        if models and len(available) < len(models):
            st.caption(
                f"{len(models) - len(available)} registered model artifact(s) are missing locally."
            )
        if st.session_state.get("previous_model") != model_id:
            for key in ("single_result", "thread_result", "batch_result", "batch_fingerprint"):
                st.session_state.pop(key, None)
            st.session_state.previous_model = model_id
        st.divider()
        st.caption("Local inference · uncalibrated scores")
        st.caption("Labels describe contributions, not factual truth.")
        st.markdown(f"[API documentation]({API_URL}/docs)")
    st.markdown('<p class="tl-kicker">DISCUSSION, IN FOCUS</p>', unsafe_allow_html=True)
    st.title("Understand the conversation.")
    st.markdown(
        '<p class="tl-subtitle">Inspect requests, supported claims, and commentary with transparent local models.</p>',
        unsafe_allow_html=True,
    )
    if selected and selected.get("provenance") == "synthetic_demo":
        st.info(
            "Synthetic demonstration model · trained on illustrative examples only. This is separate from the real-data benchmark."
        )
    if page == PAGES[0]:
        classify(model_id)
    elif page == PAGES[1]:
        thread_page(model_id)
    elif page == PAGES[2]:
        batch_page(model_id)
    elif page == PAGES[3]:
        experiment_page()
    elif page == PAGES[4]:
        review_page()
    else:
        about_page(models, selected)


if __name__ == "__main__":
    main()
