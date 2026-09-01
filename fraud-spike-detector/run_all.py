"""
One-command execution of the entire fraud-spike detection pipeline.

Usage:
    python run_all.py              # Run all stages
    python run_all.py --skip-data  # Skip data generation (reuse existing)
    python run_all.py --skip-viz   # Skip visualization generation

This runs all 5 stages in sequence:
  1. generate_data.py  -- Synthetic 60-day merchant stream + labeled fraud spikes
  2. features.py       -- Rolling-window feature engineering (16 features)
  3. detect.py         -- Train 6 detectors, generate test predictions
  4. evaluate.py       -- Precision/recall/F1, cost model, failure cases
  5. visualize.py      -- Publication-quality charts saved to reports/
"""

from __future__ import annotations

import os
import sys
import time
import logging
import argparse
import subprocess


STAGES = [
    ("1/5  Generating synthetic data...",     "src/generate_data.py", "data"),
    ("2/5  Engineering features...",           "src/features.py",     "features"),
    ("3/5  Training detectors & predicting...", "src/detect.py",      "detect"),
    ("4/5  Evaluating on held-out test set...", "src/evaluate.py",    "evaluate"),
    ("5/5  Generating visualizations...",       "src/visualize.py",   "viz"),
]


def _print_header() -> None:
    """Print the startup banner with system info."""
    print("=" * 70)
    print("SENTINELRISK-AI -- Enterprise Transaction Risk & Fraud Engine")
    print("Multi-Model Detection, Explainable AI & Financial Cost Optimization")
    print("=" * 70)
    print(f"  Python: {sys.version.split()[0]}")

    # Print key dependency versions
    try:
        import sklearn
        import pandas
        import numpy
        print(f"  NumPy: {numpy.__version__}  |  Pandas: {pandas.__version__}  |  "
              f"scikit-learn: {sklearn.__version__}")
    except ImportError:
        pass

    try:
        import xgboost
        print(f"  XGBoost: {xgboost.__version__}")
    except ImportError:
        print("  XGBoost: not installed (will use 5 detectors instead of 6)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SentinelRisk-AI pipeline")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data generation (reuse existing CSV files)")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip visualization generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    _print_header()

    total_start = time.time()
    python = sys.executable
    stage_times: list[tuple[str, float, bool]] = []  # (name, elapsed, success)

    for banner, script, stage_key in STAGES:
        # Check skip flags
        if args.skip_data and stage_key == "data":
            if os.path.exists("data/transactions_full.csv"):
                print(f"\n  [SKIP] {banner} (--skip-data, existing data found)")
                stage_times.append((banner, 0, True))
                continue
            else:
                print(f"\n  [WARN] --skip-data but no data found, generating...")

        if args.skip_viz and stage_key == "viz":
            print(f"\n  [SKIP] {banner} (--skip-viz)")
            stage_times.append((banner, 0, True))
            continue

        print()
        print("-" * 70)
        print(f"  {banner}")
        print("-" * 70)
        t0 = time.time()
        result = subprocess.run(
            [python, "-u", script],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\n  FAILED at stage: {script} (exit code {result.returncode})")
            stage_times.append((banner, elapsed, False))

            # Print which stages succeeded before failure
            print("\n  Stage Summary:")
            for name, t, ok in stage_times:
                status = "OK" if ok else "FAIL"
                print(f"    [{status}] {name} ({t:.1f}s)")
            sys.exit(1)

        stage_times.append((banner, elapsed, True))
        print(f"  Done ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print()
    print("=" * 70)
    print(f"  Pipeline complete in {total_elapsed:.1f}s")
    print("=" * 70)

    # Stage timing breakdown
    print("\n  Stage Timing:")
    for name, t, ok in stage_times:
        pct = (t / total_elapsed * 100) if total_elapsed > 0 else 0
        bar = "=" * int(pct / 3)
        skip = " [SKIPPED]" if t == 0 else ""
        print(f"    {name:<45} {t:6.1f}s ({pct:4.1f}%) {bar}{skip}")

    print()
    print("Outputs:")
    print("  data/transactions_{full,train,test}.csv  -- Raw transaction data")
    print("  data/features_{train,test}.csv           -- Window features")
    print("  data/predictions_test.csv                -- All detector predictions")
    print("  models/*.joblib                          -- Trained models")
    print("  models/metadata.json                     -- Model metadata")
    print("  reports/metrics.csv                      -- Metrics table")
    print("  reports/*.png                            -- Visualization charts")
    print()
    print("Next steps:")
    print("  streamlit run src/dashboard.py           -- Interactive dashboard")
    print("  python src/api.py                        -- REST API server")


if __name__ == "__main__":
    main()
