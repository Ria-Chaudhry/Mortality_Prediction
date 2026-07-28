from __future__ import annotations

import pandas as pd
import pytest

from clinical_domain_mortality.errors import IntegrityError
from clinical_domain_mortality.hashing import (
    hash_frame,
    hash_frame_canonical,
    hash_frame_schema,
    hash_frame_values,
    hash_object,
)
from clinical_domain_mortality.io import verify_hashes, write_csv


def test_hashes_are_deterministic():
    frame = pd.DataFrame({"b": [2, 3], "a": ["x", "y"]})
    assert hash_frame(frame) == hash_frame(frame.copy())
    assert hash_object({"b": 1, "a": 2}) == hash_object({"a": 2, "b": 1})


def test_input_output_checksum_verification(tmp_path):
    path = tmp_path / "table.csv"
    digest = write_csv(pd.DataFrame({"x": [1, 2]}), path)
    verify_hashes(tmp_path, {"table.csv": digest})
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(IntegrityError):
        verify_hashes(tmp_path, {"table.csv": digest})


def test_analytical_signature_is_order_independent_and_value_sensitive():
    frame = pd.DataFrame({"id": [1, 2], "clinical_value": [3.5, 4.5]})
    assert hash_frame_canonical(frame) == hash_frame_canonical(
        frame.iloc[::-1].reset_index(drop=True)
    )
    changed = frame.copy()
    changed.loc[0, "clinical_value"] = 9.5
    assert hash_frame_canonical(frame) != hash_frame_canonical(changed)


def test_feature_schema_and_value_hashes_have_honest_meanings():
    frame = pd.DataFrame({"cohort_visit_number": [1, 2], "feature": [0.1, 0.2]})
    changed = frame.copy()
    changed.loc[0, "feature"] = 0.3
    assert hash_frame_schema(frame) == hash_frame_schema(changed)
    assert hash_frame_values(
        frame, identity_columns=["cohort_visit_number"]
    ) != hash_frame_values(changed, identity_columns=["cohort_visit_number"])
