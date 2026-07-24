from pathlib import Path

import pandas as pd

from clinical_domains.pipeline import run_reproduction


def test_reproduction_outputs_are_deterministic(tmp_path):
    config = Path("examples/synthetic/config.yaml")
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_reproduction(config, first)
    run_reproduction(config, second)

    for artifact in ["feature_matrix.csv", "predictions.csv", "metrics.csv"]:
        left = pd.read_csv(first / artifact)
        right = pd.read_csv(second / artifact)
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)
