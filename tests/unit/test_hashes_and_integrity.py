from __future__ import annotations

import pandas as pd
import pytest

from clinical_domain_mortality.errors import IntegrityError
from clinical_domain_mortality.hashing import hash_frame, hash_object
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
