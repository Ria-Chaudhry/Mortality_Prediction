from __future__ import annotations

import pandas as pd
import pytest

from clinical_domain_mortality.audit import scan_public_tree
from clinical_domain_mortality.errors import IntegrityError
from clinical_domain_mortality.io import write_csv


def test_public_schema_rejects_row_identifiers(tmp_path):
    write_csv(
        pd.DataFrame({"patient_id": ["p1"], "probability": [0.5]}),
        tmp_path / "selected_model_performance_table.csv",
    )
    with pytest.raises(IntegrityError, match="disallowed public columns"):
        scan_public_tree(tmp_path, classification="public_synthetic")


def test_clinical_publication_requires_approval_and_small_cell_policy(tmp_path):
    write_csv(
        pd.DataFrame({"matrix": ["baseline"], "patients": [2]}),
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
