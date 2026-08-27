"""
One-command execution of the entire fraud-spike detection pipeline.

Usage:
    python run_all.py

This runs all 5 stages in sequence:
  1. generate_data.py  -- Synthetic 60-day merchant stream + labeled fraud spikes
  2. features.py       -- Rolling-window feature engineering (16 features)
  3. detect.py         -- Train 6 detectors, generate test predictions
  4. evaluate.py       -- Precision/recall/F1, cost model, failure cases
  5. visualize.py      -- Publication-quality charts saved to reports/
"""

import os
import sys
import time
import subprocess


STAGES = [
    ("1/5  Generating synthetic data...",     "src/generate_data.py"),
    ("2/5  Engineering features...",           "src/features.py"),
    ("3/5  Training detectors & predicting...", "src/detect.py"),
    ("4/5  Evaluating on held-out test set...", "src/evaluate.py"),
    ("5/5  Generating visualizations...",       "src/visualize.py"),
]


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=" * 70)
    print("FRAUD-SPIKE DETECTOR -- Full Pipeline")
    print("Built for Razorpay AI Buildathon -- Track 02: AI Risk Manager")
    print("=" * 70)

    total_start = time.time()
    python = sys.executable

    for banner, script in STAGES:
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
            print(f"\nFAILED at stage: {script}")
            sys.exit(1)
        print(f"  Done ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print()
    print("=" * 70)
    print(f"  Pipeline complete in {total_elapsed:.1f}s")
    print("=" * 70)
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
