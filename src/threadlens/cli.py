"""Reproducible local workflows. Run ``threadlens --help``."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

from threadlens.config import LABELS, settings


def emit(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def doctor() -> dict:
    versions = {}
    for name in (
        "threadlens-nlp",
        "torch",
        "transformers",
        "scikit-learn",
        "fastapi",
        "streamlit",
        "numpy",
        "pandas",
        "pytest",
        "ruff",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    try:
        import torch

        mps = torch.backends.mps.is_available()
        device = "mps" if mps else "cpu"
        reason = (
            "Apple Metal available"
            if mps
            else "MPS unavailable; CPU fallback (sandbox may hide GPU access)"
        )
    except ImportError:
        device, reason = "unavailable", "Install requirements.lock to enable neural models"
    registry = settings.artifacts / "model_registry.json"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "versions": versions,
        "device": device,
        "device_reason": reason,
        "root": str(settings.root),
        "free_disk_gib": round(shutil.disk_usage(settings.root).free / 1024**3, 2),
        "registry_present": registry.exists(),
        "labels": LABELS,
        "next_action": "threadlens serve"
        if registry.exists()
        else "threadlens demo && threadlens serve --demo; or run collect, label, split, experiments",
    }


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="ThreadLens — local NLP discussion intelligence")
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect environment and model artifacts")
    sub.add_parser("demo", help="Build isolated synthetic demo baseline; no downloads")
    run = sub.add_parser("serve", help="Start API and UI; Ctrl+C stops both")
    run.add_argument("--api-port", type=int, default=8000)
    run.add_argument("--ui-port", type=int, default=8501)
    run.add_argument("--demo", action="store_true")
    collect = sub.add_parser("collect", help="Resumable fresh official HN API collection")
    collect.add_argument("--threads", type=int, default=36)
    collect.add_argument("--per-thread", type=int, default=20)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument(
        "--manifest", type=Path, help="Replay published comment IDs and verify cleaned text hashes"
    )
    label = sub.add_parser("label", help="Resumable local NLI weak annotations")
    label.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    label.add_argument("--batch-size", type=int, default=12)
    sub.add_parser("split", help="Freeze thread-isolated 70/15/15 partitions")
    sub.add_parser("train-baselines", help="Train majority and TF-IDF on frozen train partition")
    train = sub.add_parser("train-transformer", help="Train one DistilBERT configuration/seed")
    train.add_argument("--seed", type=int, choices=(42, 7, 2024), default=42)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    experiment = sub.add_parser(
        "experiments", help="Run frozen baseline/transformer multi-seed protocol"
    )
    experiment.add_argument("--stage", choices=("initial", "improved", "all"), default="initial")
    experiment.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    sub.add_parser(
        "evaluate", help="Display recorded aggregate evaluations without rerunning tuning"
    )
    report = sub.add_parser("report", help="Export recorded experiment report as Markdown")
    report.add_argument("--output", type=Path, default=settings.reports / "results.md")
    predict = sub.add_parser("predict", help="Classify using local artifacts")
    predict.add_argument("text")
    predict.add_argument("--model", default=None)
    return cli


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "doctor":
            emit(doctor())
        elif args.command == "demo":
            from threadlens.demo import prepare_demo

            emit(prepare_demo(settings.local / "demo"))
        elif args.command == "serve":
            from threadlens.launcher import serve

            serve(args.api_port, args.ui_port, args.demo)
        elif args.command == "collect":
            from threadlens.data.hn import collect_dataset

            if args.manifest:
                from threadlens.data.replay import replay_manifest

                emit(replay_manifest(args.manifest))
            else:
                emit(
                    collect_dataset(
                        threads=args.threads, per_thread=args.per_thread, seed=args.seed
                    )
                )
        elif args.command == "label":
            from threadlens.labeling.weak import label_dataset

            emit(label_dataset(device=args.device, batch_size=args.batch_size))
        elif args.command == "split":
            from threadlens.data.splits import freeze_splits

            records = [
                json.loads(line)
                for line in (settings.local / "labeled.jsonl").read_text().splitlines()
                if line
            ]
            emit(freeze_splits(records))
        elif args.command in ("evaluate", "report"):
            path = settings.reports / "experiments.json"
            if not path.exists():
                raise ValueError("No recorded experiments. Run threadlens experiments first.")
            report = json.loads(path.read_text())
            if args.command == "evaluate":
                emit(report)
            else:
                from threadlens.evaluation.reporting import generate_reports

                generate_reports()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text((settings.root / "docs/experiments.md").read_text())
                print(args.output)
        elif args.command == "predict":
            from threadlens.models.inference import ModelManager

            emit(ModelManager(settings).predict([args.text], args.model)[0])
        else:
            from threadlens.data.splits import freeze_splits
            from threadlens.models import training

            if args.command == "train-baselines":
                records = training.read_labeled()
                emit(training.train_baselines(records, freeze_splits(records)))
            elif args.command == "train-transformer":
                emit(
                    training.train_transformer(
                        seed=args.seed, epochs=args.epochs, device=args.device
                    )
                )
            elif args.command == "experiments":
                from threadlens.evaluation.runner import run_experiments

                emit(run_experiments(stage=args.stage, device=args.device))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ThreadLens: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
