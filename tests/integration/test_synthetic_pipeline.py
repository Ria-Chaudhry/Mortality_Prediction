from __future__ import annotations

import pandas as pd
import pytest

from clinical_domain_mortality.config import PROJECT_ROOT
from clinical_domain_mortality.pipeline import run_pipeline, verify_run


@pytest.mark.slow
def test_both_adapters_end_to_end(tmp_path):
    public = tmp_path / "public"
    restricted = tmp_path / "restricted"
    chorus = run_pipeline(
        PROJECT_ROOT / "configs" / "chorus.example.yaml", public, restricted
    )
    mimic = run_pipeline(
        PROJECT_ROOT / "configs" / "mimic.example.yaml", public, restricted
    )
    assert chorus.run_manifest["dataset"] == "chorus"
    assert mimic.run_manifest["dataset"] == "mimiciv"
    for dataset in ("chorus", "mimiciv"):
        predictions = pd.read_csv(
            restricted / dataset / "oof_predictions_restricted.csv"
        )
        assert len(predictions) == 70 * 32
        assert not predictions.duplicated(
            ["cohort_visit_number", "matrix", "model"]
        ).any()
        assert predictions["probability"].between(0, 1).all()
        verify_run(public / dataset)
