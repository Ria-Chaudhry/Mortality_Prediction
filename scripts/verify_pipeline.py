from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_domains.core.validation import validate_records_against_schema
from clinical_domains.pipeline import run_reproduction
from clinical_domains.utils.config import load_yaml


def _validate_run(output_dir: Path) -> None:
    required = {
        "feature_matrix": output_dir / "feature_matrix.csv",
        "predictions": output_dir / "predictions.csv",
        "metrics": output_dir / "metrics.csv",
        "manuscript_metrics": output_dir / "manuscript_metrics.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing expected output artifacts: {missing}")

    predictions = pd.read_csv(required["predictions"])
    metrics = pd.read_csv(required["metrics"])

    validate_records_against_schema(predictions, "schemas/prediction_schema.json")
    validate_records_against_schema(metrics, "schemas/output_schema.json")

    if not predictions["y_score"].between(0, 1).all():
        raise SystemExit("Predicted probabilities must be between 0 and 1.")
    if not np.isfinite(metrics["estimate"]).all():
        raise SystemExit("Metric estimates must be finite.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/synthetic/config.yaml")
    parser.add_argument("--output-dir", default="outputs/verification")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if config.get("dataset", {}).get("name") != "synthetic":
        raise SystemExit("Verification is intended for the committed synthetic dataset.")

    output_root = Path(args.output_dir)
    first = output_root / "run_1"
    second = output_root / "run_2"
    run_reproduction(args.config, first)
    run_reproduction(args.config, second)
    _validate_run(first)
    _validate_run(second)

    for artifact in ["feature_matrix.csv", "predictions.csv", "metrics.csv"]:
        left = pd.read_csv(first / artifact)
        right = pd.read_csv(second / artifact)
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)

    print(f"Verified reproducible synthetic pipeline outputs in {output_root}")


if __name__ == "__main__":
    main()
