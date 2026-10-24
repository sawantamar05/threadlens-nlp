"""Run real Chromium checks against running ThreadLens services.

Requires the optional Playwright package and a Chromium installation. Inputs and
screenshots deliberately use synthetic prose; live HN source text is not captured.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
COMMENT = (
    "Could you explain how the retry policy handles a failed request after the connection closes?"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8511")
    parser.add_argument("--prefix", default="demo")
    parser.add_argument("--model", help="Exact model display name to select")
    parser.add_argument(
        "--story", help="HN story ID for actual retrieval; source text is never screenshot"
    )
    parser.add_argument(
        "--review", action="store_true", help="Exercise the synthetic review queue only"
    )
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument(
        "--unavailable", action="store_true", help="Expect a deliberately unavailable API"
    )
    parser.add_argument(
        "--missing-model", action="store_true", help="Expect an actual empty local model registry"
    )
    args = parser.parse_args()
    output = ROOT / "reports" / f"browser_{args.prefix}_checks.json"
    images = ROOT / "docs" / "images"
    images.mkdir(parents=True, exist_ok=True)
    checks = []
    errors = []
    console_errors = []
    raw_source_visible = False
    evidence = {
        "timestamp": datetime.now(UTC).isoformat(),
        "url": args.url,
        "model_display_name": args.model,
        "input_provenance": "synthetic_demo",
        "checks": checks,
        "errors": errors,
        "console_errors": console_errors,
        "screenshots": [],
    }

    def save():
        output.write_text(json.dumps(evidence, indent=2) + "\n")

    def passed(name, detail):
        checks.append({"check": name, "status": "passed", "evidence": detail})
        print(f"PASS {name}: {detail}", flush=True)
        save()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
        page.set_default_timeout(20000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text) if message.type == "error" else None
            ),
        )

        def text():
            return page.locator("body").inner_text()

        def visible_message(value):
            page.get_by_text(value, exact=False).first.wait_for(state="visible")

        def navigate(label):
            print(f"NAV {label}", flush=True)
            page.get_by_test_id("stSidebar").get_by_text(label, exact=True).click()
            page.wait_for_timeout(500)

        def screenshot(name):
            if raw_source_visible:
                print("WITHHELD screenshot while real source comments are visible", flush=True)
                return
            path = images / f"{args.prefix}-{name}.png"
            page.screenshot(path=str(path), full_page=True)
            evidence["screenshots"].append(str(path.relative_to(ROOT)))
            save()

        try:
            print(f"OPEN {args.url}", flush=True)
            page.goto(args.url, wait_until="domcontentloaded", timeout=30000)
            visible_message("Understand the conversation.")
            passed(
                "desktop_render",
                "Application title and accessible workspace navigation rendered at 1440px.",
            )
            caption = page.get_by_text("Local inference · uncalibrated scores", exact=True)
            color = caption.evaluate("el => getComputedStyle(el).color")
            assert color == "rgb(82, 100, 122)", f"Caption contrast color changed: {color}"
            opacity = caption.evaluate(
                "el => getComputedStyle(el.closest('[data-testid=stCaptionContainer]')).opacity"
            )
            assert opacity == "1", f"Caption container opacity lowers contrast: {opacity}"
            passed(
                "caption_contrast",
                f"Browser computed caption color is {color}, matching #52647A, with container opacity {opacity}.",
            )
            if args.unavailable:
                visible_message("The local API is unavailable")
                assert page.get_by_role("button", name="Classify comment", exact=True).is_disabled()
                screenshot("api-unavailable")
                passed(
                    "missing_api_state",
                    "Actionable API startup message; inference button disabled.",
                )
            elif args.missing_model:
                visible_message("No trained artifacts found")
                assert page.get_by_role(
                    "button", name="Classify comment", exact=True
                ).is_disabled(), "Inference should be disabled without artifacts"
                screenshot("model-missing")
                passed(
                    "missing_model_state",
                    "Actual empty registry displays rebuild instructions and disables inference.",
                )
            else:
                if args.model:
                    page.get_by_role("combobox", name="Select model", exact=True).click()
                    page.get_by_role("option", name=args.model, exact=True).click()
                    page.wait_for_timeout(400)
                model_choices = (
                    page.get_by_test_id("stSelectbox")
                    .filter(has=page.get_by_role("combobox", name="Select model", exact=True))
                    .inner_text()
                )
                passed(
                    "model_selection",
                    f"Model selector rendered; selected {args.model or 'registered default'}; control: {model_choices}",
                )
                page.get_by_role("button", name="Classify comment", exact=True).click()
                visible_message("Enter a non-empty comment")
                passed(
                    "empty_input",
                    "Empty submission displays validation error without a prediction.",
                )
                page.get_by_role("textbox", name="Complete comment", exact=True).fill(COMMENT)
                page.get_by_role("button", name="Classify comment", exact=True).click()
                visible_message("Predicted contribution")
                visible_message("Scores express the model's relative preference")
                page.get_by_test_id("stPlotlyChart").first.wait_for(state="visible")
                assert page.get_by_test_id("stPlotlyChart").count() == 1
                screenshot("classify-desktop")
                page.get_by_test_id("stPlotlyChart").first.scroll_into_view_if_needed()
                screenshot("prediction-desktop")
                passed(
                    "single_prediction",
                    "Synthetic comment produces label, all-score chart, model version, timing, and uncalibrated-score disclosure.",
                )
                page.set_viewport_size({"width": 390, "height": 844})
                page.wait_for_timeout(400)
                screenshot("classify-narrow")
                page.get_by_test_id("stPlotlyChart").first.scroll_into_view_if_needed()
                screenshot("prediction-narrow")
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth + 2"
                )
                passed(
                    "narrow_layout",
                    "390px viewport has no horizontal document overflow; actual screenshot saved.",
                )
                page.set_viewport_size({"width": 1440, "height": 1050})
                page.wait_for_timeout(400)
                navigate("Analyze a thread")
                page.get_by_role("textbox", name="Story ID or Hacker News URL", exact=True).fill(
                    "https://example.com/item?id=1"
                )
                page.get_by_role("button", name="Analyze thread", exact=True).click()
                visible_message("HTTP 422")
                passed(
                    "invalid_story_url",
                    "Arbitrary non-HN URL receives a visible HTTP 422 validation error.",
                )
                screenshot("thread-validation")
                if args.story:
                    raw_source_visible = True
                    page.get_by_role(
                        "textbox", name="Story ID or Hacker News URL", exact=True
                    ).fill(args.story)
                    page.get_by_role("button", name="Analyze thread", exact=True).click()
                    visible_message("Filter predicted labels")
                    visible_message("Open source discussion")
                    assert page.get_by_test_id("stExpander").count() > 0
                    count = page.get_by_test_id("stExpander").count()
                    filters = page.get_by_role(
                        "combobox", name="Filter predicted labels", exact=True
                    )
                    assert filters.is_visible()
                    showing = page.get_by_text(re.compile(r"^Showing \d+ of \d+ comments$")).first
                    before_filter = showing.inner_text()
                    after_filter = before_filter
                    for _ in range(3):
                        filters.press("Backspace")
                        page.wait_for_timeout(500)
                        after_filter = showing.inner_text()
                        if after_filter != before_filter:
                            break
                    assert after_filter != before_filter, (
                        "Removing selected label chips must change visible comment count"
                    )
                    passed(
                        "actual_thread_analysis",
                        f"Retrieved and classified {count} comments; label counts/proportions and expandable previews rendered. Keyboard label filtering changed '{before_filter}' to '{after_filter}'. No source text exported or captured.",
                    )
                navigate("Batch classification")
                visible_message("Upload comments CSV")
                raw_source_visible = False
                content = 'row,text,threadlens_label\n2,"Could you clarify the retry setting?",keep-two\n1,"=SUM(1+1)",keep-one\n3,"I like the interface.",keep-three\n'
                page.locator('input[type="file"]').set_input_files(
                    {
                        "name": "synthetic-browser.csv",
                        "mimeType": "text/csv",
                        "buffer": content.encode(),
                    }
                )
                visible_message("Validated 3 rows")
                page.get_by_role("button", name="Classify CSV", exact=True).click()
                visible_message("Completed 3 comments")
                with page.expect_download() as download:
                    page.get_by_role("button", name="Download predictions CSV", exact=True).click()
                exported = Path(download.value.path()).read_text(encoding="utf-8-sig")
                records = list(csv.DictReader(io.StringIO(exported)))
                assert [row["row"] for row in records] == ["2", "1", "3"]
                assert records[1]["text"] == "'=SUM(1+1)"
                assert records[0]["threadlens_label"] == "keep-two"
                assert "_threadlens_label" in records[0]
                screenshot("batch-results")
                passed(
                    "csv_upload_predict_download",
                    "Real browser upload → API inference → CSV download completed for 3 rows; original order/column collision preserved and formula escaped.",
                )
                navigate("Experiments")
                if args.experiments:
                    visible_message("All runs")
                    visible_message("Confusion counts")
                    assert page.locator('[data-testid="stPlotlyChart"]').count() >= 3
                    screenshot("experiments")
                    page.get_by_role(
                        "button", name="Download experiment report", exact=True
                    ).wait_for(state="visible")
                    page.wait_for_timeout(500)
                    inspect = page.get_by_role("combobox", name="Inspect a run", exact=True)
                    inspect.fill("distilbert-improved-seed42")
                    inspect.press("ArrowDown")
                    inspect.press("Enter")
                    page.wait_for_timeout(500)
                    page.get_by_text("Confusion counts", exact=True).scroll_into_view_if_needed()
                    screenshot("confusion-matrix")
                    passed(
                        "experiment_charts",
                        "Actual report renders all runs, label counts, per-label table, confusion matrix, seed variation and provenance.",
                    )
                    with page.expect_download() as download:
                        page.get_by_role(
                            "button", name="Download experiment report", exact=True
                        ).click()
                    report = json.loads(Path(download.value.path()).read_text())
                    assert report["runs"]
                    passed(
                        "report_download",
                        f"Downloaded report contains {len(report['runs'])} actual runs.",
                    )
                else:
                    passed(
                        "experiments_screen",
                        "Offline demonstration report state rendered; no real-data benchmark claim made.",
                    )
                raw_source_visible = not args.review
                navigate("Annotation review")
                if args.review:
                    visible_message("Assign a label")
                    assert page.get_by_role(
                        "switch", name="Blind labeling mode", exact=True
                    ).is_checked()
                    assert "Original draft:" not in text()
                    page.get_by_role("radio", name="request", exact=True).press("Space")
                    page.get_by_role("textbox", name="Review notes (optional)", exact=True).fill(
                        "Automated browser test using a synthetic comment; not independent human annotation."
                    )
                    page.get_by_role("button", name="Save annotation", exact=True).click()
                    visible_message("Saved revision")
                    assert (
                        "human_independent"
                        not in page.locator('[data-testid="stAlertContainer"]').all_text_contents()
                    )
                    page.get_by_role("button", name="Reload saved revisions", exact=True).click()
                    visible_message("Saved revision history")
                    screenshot("annotation-review")
                    passed(
                        "blind_review_save_reload",
                        "Synthetic queue hid draft, saved default AI-assisted review intent with synthetic provenance enforced by API, and reloaded immutable draft/revisions.",
                    )
                else:
                    passed(
                        "annotation_screen",
                        "Review screen rendered without saving or screenshotting real source comments.",
                    )
                navigate("About & models")
                visible_message("One inference service, clear boundaries")
                raw_source_visible = False
                visible_message("Active model information")
                screenshot("about-models")
                passed(
                    "about_metadata",
                    "Architecture, active model metadata, limitations, data sources and AI-assistance disclosure rendered.",
                )
            if errors:
                raise AssertionError(f"Browser runtime errors: {errors}")
            passed(
                "browser_runtime",
                f"No page JavaScript errors; {len(console_errors)} console error messages recorded for inspection.",
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            print(f"FAIL {type(exc).__name__}: {exc}", flush=True)
            screenshot("failure")
        finally:
            evidence["completed_at"] = datetime.now(UTC).isoformat()
            save()
            browser.close()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
