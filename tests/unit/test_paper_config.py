from __future__ import annotations

import shutil
from copy import deepcopy

import pandas as pd
import pytest

from clinical_domain_mortality.config import PROJECT_ROOT, load_config, validate_config
from clinical_domain_mortality.errors import ConfigurationError, IntegrityError
from clinical_domain_mortality.pipeline import (
    _verify_selection_artifacts,
    paper_preflight,
    run_pipeline,
)
from clinical_domain_mortality.runtime import (
    frozen_verification_runtime_supported,
    validate_frozen_verification_runtime,
    validate_supported_runtime,
)


def test_unconfirmed_chorus_paper_config_fails_closed():
    with pytest.raises(ConfigurationError, match="fail-closed"):
        load_config(PROJECT_ROOT / "configs" / "chorus.paper.yaml")


def test_chorus_paper_preflight_reports_blocker_without_source_access():
    result = paper_preflight(PROJECT_ROOT / "configs" / "chorus.paper.yaml")
    assert result["status"] == "blocked"
    assert result["source_access_attempted"] is False
    assert "fail-closed" in result["reason"]


def test_recovered_mimic_paper_config_passes_source_free_preflight():
    config = load_config(PROJECT_ROOT / "configs" / "mimiciv.paper.yaml")
    assert config["source"]["release_or_snapshot"] == "v3.1"
    assert config["cohort"]["min_age_years"] == 0
    assert config["cohort"]["row_order_policy"] == "patient_start_visit_v1"
    assert (
        config["cohort"]["predictor_window_end_policy"]
        == "admission_plus_window_v1"
    )
    assert (
        config["folds"]["method"]
        == "historical_stratified_group_k_fold_v1"
    )
    assert config["cohort"]["expected_counts"] == {
        "visits": 23000,
        "patients": 10006,
        "deaths": 819,
    }
    assert config["features"]["concept_count"] == 50
    assert config["features"]["retained_derived_feature_count"] == 21
    result = paper_preflight(PROJECT_ROOT / "configs" / "mimiciv.paper.yaml")
    assert result["status"] == "ready"
    assert result["source_access_attempted"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("features", "concept_count"), 250, "50 concepts"),
        (("features", "retained_derived_feature_count"), 15, "21 retained"),
        (("features", "measurements", "expected_count"), 15, "expected_count=21"),
        (("features", "medications", "expected_count"), 15, "expected_count=21"),
        (("features", "procedures", "expected_count"), 15, "expected_count=21"),
    ],
)
def test_mimic_selection_count_regressions_fail_hard(path, value, message):
    config = deepcopy(load_config(PROJECT_ROOT / "configs" / "mimic.example.yaml"))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigurationError, match=message):
        validate_config(config)


def test_unsupported_python_runtime_fails_before_execution(monkeypatch):
    monkeypatch.setattr("platform.python_version", lambda: "3.12.13")
    with pytest.raises(ConfigurationError, match="CPython 3.10.13"):
        validate_supported_runtime()


def test_frozen_verification_runtime_is_fail_closed(monkeypatch):
    monkeypatch.setattr("platform.python_version", lambda: "3.10.13")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    assert frozen_verification_runtime_supported() is False
    with pytest.raises(ConfigurationError, match="Linux x86_64"):
        validate_frozen_verification_runtime()


def test_paper_selection_verification_reads_actual_tables(tmp_path, chorus_config):
    result = run_pipeline(
        chorus_config["_meta"]["source_config"],
        output_base=tmp_path / "public",
        restricted_base=tmp_path / "restricted",
        stop_after=6,
    )
    for name in (
        "fold_concept_selections.csv",
        "fold_derived_feature_selections.csv",
        "fold_selection_audit.csv",
    ):
        shutil.copy2(result.public_dir / name, result.restricted_dir / name)
    _verify_selection_artifacts(chorus_config, result.restricted_dir)
    derived_path = result.restricted_dir / "fold_derived_feature_selections.csv"
    original_derived = pd.read_csv(derived_path)
    derived_columns = [
        "fold",
        "domain",
        "rank",
        "candidate_feature_name",
        "source_concept",
        "summary_type",
        "training_support_count",
        "training_support_proportion",
        "selection_score",
        "tie_break_value",
        "training_visit_count",
        "support_definition",
        "selected",
        "selection_rule_identifier",
        "selection_rule_version",
        "eligibility_status",
        "derived_selection_hash",
    ]
    for column in derived_columns:
        corrupted = original_derived.copy()
        value = corrupted.loc[0, column]
        corrupted.loc[0, column] = (
            not bool(value)
            if column == "selected"
            else value + 1
            if pd.api.types.is_number(value)
            else f"{value}-corrupted"
        )
        corrupted.to_csv(derived_path, index=False)
        with pytest.raises(IntegrityError):
            _verify_selection_artifacts(chorus_config, result.restricted_dir)
    original_derived.to_csv(derived_path, index=False)

    concept_path = result.restricted_dir / "fold_concept_selections.csv"
    original_concepts = pd.read_csv(concept_path)
    concept_columns = [
        "fold",
        "domain",
        "rank",
        "concept_key",
        "concept_name",
        "training_visit_prevalence",
        "source_table",
        "semantics",
        "units",
        "selected",
        "selection_hash",
    ]
    for column in concept_columns:
        corrupted = original_concepts.copy()
        value = corrupted.loc[0, column]
        corrupted.loc[0, column] = (
            not bool(value)
            if column == "selected"
            else value + 1
            if pd.api.types.is_number(value)
            else f"{value}-corrupted"
        )
        corrupted.to_csv(concept_path, index=False)
        with pytest.raises(IntegrityError):
            _verify_selection_artifacts(chorus_config, result.restricted_dir)
    original_concepts.to_csv(concept_path, index=False)

    combined_path = result.restricted_dir / "fold_selection_audit.csv"
    original_combined = pd.read_csv(combined_path)
    corrupted_combined = original_combined.copy()
    corrupted_combined.loc[0, "candidate_feature_rank"] += 1
    corrupted_combined.to_csv(combined_path, index=False)
    with pytest.raises(IntegrityError, match="combined selection audit"):
        _verify_selection_artifacts(chorus_config, result.restricted_dir)
    original_combined.to_csv(combined_path, index=False)
