from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from clinical_domain_mortality.audit import scan_public_tree
from clinical_domain_mortality.audit.privacy import (
    PUBLIC_CLINICAL_JSON_SCHEMAS,
    PUBLIC_CLINICAL_TABLE_SCHEMAS,
)
from clinical_domain_mortality.config import PROJECT_ROOT
from clinical_domain_mortality.errors import ConfigurationError, IntegrityError
from clinical_domain_mortality.hashing import hash_file
from clinical_domain_mortality.io import read_json, write_csv, write_json
from clinical_domain_mortality.pipeline import verify_run
from clinical_domain_mortality.runtime import (
    frozen_verification_runtime_supported,
)


def _run_in_fresh_process(config, public, restricted):
    environment = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "clinical_domain_mortality",
            "run",
            "--config",
            str(config),
            "--output-dir",
            str(public),
            "--restricted-output-dir",
            str(restricted),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert '"status": "ok"' in completed.stdout


@pytest.mark.slow
def test_both_adapters_end_to_end(tmp_path):
    public = tmp_path / "public"
    restricted = tmp_path / "restricted"
    repeated_public = tmp_path / "repeated_public"
    repeated_restricted = tmp_path / "repeated_restricted"
    _run_in_fresh_process(
        PROJECT_ROOT / "configs" / "chorus.example.yaml",
        public,
        restricted,
    )
    _run_in_fresh_process(
        PROJECT_ROOT / "configs" / "mimic.example.yaml",
        public,
        restricted,
    )
    _run_in_fresh_process(
        PROJECT_ROOT / "configs" / "chorus.example.yaml",
        repeated_public,
        repeated_restricted,
    )
    chorus = read_json(public / "chorus" / "run_manifest.json")
    mimic = read_json(public / "mimiciv" / "run_manifest.json")
    repeated_chorus = read_json(
        repeated_public / "chorus" / "run_manifest.json"
    )
    assert (
        repeated_chorus["output_hashes"]
        == chorus["output_hashes"]
    )
    assert repeated_chorus["run_id"] == chorus["run_id"]
    assert chorus["dataset"] == "chorus"
    assert mimic["dataset"] == "mimiciv"
    for dataset in ("chorus", "mimiciv"):
        predictions = pd.read_csv(
            restricted / dataset / "oof_predictions_restricted.csv"
        )
        assert len(predictions) == 70 * 32
        assert not predictions.duplicated(
            ["cohort_visit_number", "matrix", "model"]
        ).any()
        assert predictions["probability"].between(0, 1).all()
        shap_summary = pd.read_csv(public / dataset / "shap_summary.csv")
        shap_folds = pd.read_csv(
            public / dataset / "shap_fold_aggregates.csv"
        )
        best_models = pd.read_csv(
            public / dataset / "best_model_by_matrix.csv"
        ).set_index("matrix")["model"]
        assert shap_summary["mean_absolute_shap"].ge(0).all()
        assert not shap_summary.duplicated(
            ["feature_matrix", "feature"]
        ).any()
        assert set(shap_folds["outer_fold"]) == set(range(5))
        assert set(shap_folds["evaluation_partition"]) == {
            "outer_validation_fold"
        }
        for matrix, model in best_models.items():
            assert set(
                shap_folds.loc[
                    shap_folds["feature_matrix"].eq(matrix), "model"
                ]
            ) == {model}
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
                "chorus": chorus["run_id"],
                "mimiciv": mimic["run_id"],
            },
            "summary_hash": hash_file(public / "synthetic_run_summary.csv"),
        },
        public / "run_manifest.json",
    )
    if frozen_verification_runtime_supported():
        verify_run(public)
    else:
        with pytest.raises(
            ConfigurationError,
            match="Exact frozen fitted-model verification supports only",
        ):
            verify_run(public)

    clinical_candidate = tmp_path / "clinical_candidate"
    for source in (public / "chorus").rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(public / "chorus").as_posix()
        if (
            relative in PUBLIC_CLINICAL_TABLE_SCHEMAS
            or relative in PUBLIC_CLINICAL_JSON_SCHEMAS
        ):
            target = clinical_candidate / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    scan_public_tree(
        clinical_candidate,
        classification="public_clinical",
        small_cell_threshold=1,
        release_approved=True,
    )
    unexpected = public / "chorus" / "unexpected_aggregate.csv"
    unexpected.write_text("metric,value\nx,1\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="artifact set"):
        verify_run(public / "chorus")
    unexpected.unlink()

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
        "shap_summary.csv",
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
    with pytest.raises(IntegrityError, match="Run ID is inconsistent"):
        verify_run(public)
    write_json(original_manifest, manifest_path)

    changed_manifest = dict(original_manifest)
    changed_manifest["code_hash"] = "adversarial-change"
    write_json(changed_manifest, manifest_path)
    with pytest.raises(IntegrityError, match="code hash"):
        verify_run(public)
    write_json(original_manifest, manifest_path)
