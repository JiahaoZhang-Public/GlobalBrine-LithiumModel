#!/usr/bin/env python3
"""Orchestrator: runs compute then plot in sequence.

Usage:
    python evaluations/run_evaluation.py          # full pipeline
    python evaluations/run_evaluation.py --plots-only  # regenerate plots from metadata
"""
from __future__ import annotations

import argparse
import sys

from run_compute import main as compute_main
from run_plots import main as plot_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline.")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip compute, regenerate plots from existing metadata.",
    )
    args, remaining = parser.parse_known_args()

    # Forward remaining args to sub-scripts via sys.argv patching
    if args.plots_only:
        sys.argv = ["run_plots.py"] + remaining
        plot_main()
    else:
        sys.argv = ["run_compute.py"] + remaining
        compute_main()
        sys.argv = ["run_plots.py"]
        plot_main()


if __name__ == "__main__":
    main()
