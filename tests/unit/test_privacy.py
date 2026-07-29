from __future__ import annotations

import pandas as pd
import pytest

from clinical_domain_mortality.audit import scan_public_tree
from clinical_domain_mortality.audit.privacy import PUBLIC_CLINICAL_TABLE_SCHEMAS
from clinical_domain_mortality.errors import IntegrityError
from clinical_domain_mortality.io import write_csv, write_json


def _schema_row(filename, **updates):
    columns = PUBLIC_CLINICAL_TABLE_SCHEMAS[filename]
    row = {column: 0 for column in columns}
    row.update(updates)
    return pd.DataFrame([row], columns=columns)


def test_public_schema_rejects_row_identifiers(tmp_path):
    write_csv(
        pd.DataFrame({"patient_id": ["p1"], "probability": [0.5]}),
        tmp_path / "selected_model_performance_table.csv",
    )
    with pytest.raises(IntegrityError, match="disallowed public columns"):
        scan_public_tree(tmp_path, classification="public_synthetic")


def test_clinical_publication_requires_approval_and_small_cell_policy(tmp_path):
    write_csv(
        _schema_row(
            "selected_model_performance_table.csv",
            matrix="baseline",
            model="logistic_regression",
            visits=2,
        ),
        tmp_path / "selected_model_performance_table.csv",
    )
    with pytest.raises(IntegrityError, match="release approval"):
        scan_public_tree(
            tmp_path,
            classification="public_clinical",
            small_cell_threshold=5,
            release_approved=False,
        )
    with pytest.raises(IntegrityError, match="small cell"):
        scan_public_tree(
            tmp_path,
            classification="public_clinical",
            small_cell_threshold=5,
            release_approved=True,
        )


def test_measurement_unit_audit_is_public_only_for_synthetic_by_default(tmp_path):
    write_csv(
        pd.DataFrame(
            {
                "fold": [0],
                "concept_key": ["synthetic"],
                "unit": ["u"],
                "status": ["valid"],
                "count": [1],
            }
        ),
        tmp_path / "measurement_unit_audit.csv",
    )
    scan_public_tree(tmp_path, classification="public_synthetic")
    scan_public_tree(tmp_path, classification="release_candidate_aggregate")
    with pytest.raises(IntegrityError, match="allowlist|small cell"):
        scan_public_tree(
            tmp_path,
            classification="public_clinical",
            small_cell_threshold=5,
            release_approved=True,
        )


def test_approval_flag_cannot_bypass_public_clinical_small_cell_gate(tmp_path):
    write_csv(
        _schema_row(
            "selected_model_performance_table.csv",
            matrix="baseline",
            model="logistic_regression",
            visits=1,
        ),
        tmp_path / "selected_model_performance_table.csv",
    )
    with pytest.raises(IntegrityError, match="small cell"):
        scan_public_tree(
            tmp_path,
            classification="public_clinical",
            small_cell_threshold=5,
            release_approved=True,
        )


@pytest.mark.parametrize(
    "column",
    [
        "person_id",
        "case_identifier",
        "encounter_number",
        "event_datetime",
        "oof_probability",
        "feature_value",
    ],
)
def test_public_scanner_rejects_identifier_date_and_row_level_aliases(
    tmp_path, column
):
    path = tmp_path / f"{column}.parquet"
    pd.DataFrame({column: [1]}).to_parquet(path, index=False)
    with pytest.raises(IntegrityError, match="disallowed public columns"):
        scan_public_tree(tmp_path, classification="public_synthetic")


def test_public_clinical_rejects_unallowlisted_file_and_nested_identifier(
    tmp_path,
):
    write_json(
        {"summary": {"subject_id": "renamed-value"}},
        tmp_path / "unreviewed_summary.json",
    )
    with pytest.raises(IntegrityError, match="allowlist|disallowed public JSON"):
        scan_public_tree(
            tmp_path,
            classification="public_clinical",
            small_cell_threshold=5,
            release_approved=True,
        )


def test_public_clinical_small_cell_gate_covers_clinical_utility_counts(
    tmp_path,
):
    filename = "selected_model_clinical_utility_table.csv"
    write_csv(
        _schema_row(
            filename,
            matrix="baseline",
            model="logistic_regression",
            deaths_captured=1,
            total_deaths=10,
        ),
        tmp_path / filename,
    )
    with pytest.raises(IntegrityError, match="small cell"):
        scan_public_tree(
            tmp_path,
            classification="public_clinical",
            small_cell_threshold=5,
            release_approved=True,
        )


def test_legitimate_allowlisted_clinical_aggregate_passes(tmp_path):
    filename = "best_model_by_matrix.csv"
    write_csv(
        _schema_row(
            filename,
            matrix="baseline",
            model="logistic_regression",
            selection_rule="auprc_desc_auroc_desc_brier_asc_model_order",
        ),
        tmp_path / filename,
    )
    scan_public_tree(
        tmp_path,
        classification="public_clinical",
        small_cell_threshold=5,
        release_approved=True,
    )
