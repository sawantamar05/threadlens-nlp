"""Explicit stages preserve an auditable declaration before improved test evaluation."""

import argparse

from threadlens.evaluation.runner import run_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["initial", "improved", "all"], default="initial")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    run_experiments(stage=args.stage, device=args.device)
