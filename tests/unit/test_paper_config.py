from __future__ import annotations

import shutil

import pandas as pd
import pytest

from clinical_domain_mortality.config import PROJECT_ROOT, load_config
from clinical_domain_mortality.errors import ConfigurationError, IntegrityError
from clinical_domain_mortality.pipeline import (
    _verify_selection_artifacts,
    paper_preflight,
    run_pipeline,
)
from clinical_domain_mortality.runtime import validate_supported_runtime


@pytest.mark.parametrize(
    "name", ["chorus.paper.yaml", "mimiciv.paper.yaml"]
)
def test_unconfirmed_paper_configs_fail_closed(name):
    with pytest.raises(ConfigurationError, match="fail-closed"):
        load_config(PROJECT_ROOT / "configs" / name)


@pytest.mark.parametrize(
    "name", ["chorus.paper.yaml", "mimiciv.paper.yaml"]
)
def test_paper_preflight_reports_blocker_without_source_access(name):
    result = paper_preflight(PROJECT_ROOT / "configs" / name)
    assert result["status"] == "blocked"
    assert result["source_access_attempted"] is False
    assert "fail-closed" in result["reason"]


def test_unsupported_python_runtime_fails_before_execution(monkeypatch):
    monkeypatch.setattr("platform.python_version", lambda: "3.12.13")
    with pytest.raises(ConfigurationError, match="CPython 3.10.13"):
        validate_supported_runtime()


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
