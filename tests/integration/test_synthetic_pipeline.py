from __future__ import annotations

import pandas as pd
import pytest

from clinical_domain_mortality.config import PROJECT_ROOT
from clinical_domain_mortality.errors import IntegrityError
from clinical_domain_mortality.hashing import hash_file
from clinical_domain_mortality.io import read_json, write_csv, write_json
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
    summary = pd.DataFrame(
        [
            read_json(public / dataset / "manifests" / "dataset_manifest.json")
            for dataset in ("chorus", "mimiciv")
        ]
    ).sort_values("dataset", kind="stable")
    write_csv(summary, public / "synthetic_run_summary.csv")
    write_json(
        {
            "created_utc": "excluded-from-canonical-verification",
            "datasets": ["chorus", "mimiciv"],
            "child_run_ids": {
                "chorus": chorus.run_manifest["run_id"],
                "mimiciv": mimic.run_manifest["run_id"],
            },
            "summary_hash": hash_file(public / "synthetic_run_summary.csv"),
        },
        public / "run_manifest.json",
    )
    verify_run(public)

    major_classes = [
        "pooled_oof_metrics.csv",
        "fold_metrics.csv",
        "best_model_by_matrix.csv",
        "selected_models_calibration_coordinates.csv",
        "selected_models_sensitivity_at_90_specificity.csv",
        "selected_models_top_10_percent_risk_analysis.csv",
        "selected_model_clinical_utility_table.csv",
        "selected_models_decision_curve_coordinates.csv",
        "prespecified_paired_matrix_comparisons.csv",
        "attrition.csv",
        "fold_derived_feature_selections.csv",
    ]
    for relative in major_classes:
        target = public / "chorus" / relative
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        with pytest.raises(IntegrityError):
            verify_run(public)
        target.write_bytes(original)

    manifest_path = public / "chorus" / "run_manifest.json"
    original_manifest = read_json(manifest_path)
    changed_manifest = dict(original_manifest)
    changed_manifest["run_id"] = "adversarial-change"
    write_json(changed_manifest, manifest_path)
    with pytest.raises(IntegrityError):
        verify_run(public)
    write_json(original_manifest, manifest_path)
