from __future__ import annotations

import inspect

from clinical_domain_mortality import io
from clinical_domain_mortality.adapters import base, chorus


def test_chorus_adapter_never_issues_select_star():
    source = inspect.getsource(chorus.CHoRUSAdapter)
    assert "SELECT *" not in source.upper()
    assert "cdm_candidate_acute_cohort" in source
    assert "EXISTS (SELECT 1 FROM" in source
    assert "expanding=True" not in source
    assert " IN :eligible" not in source
    assert "predictor_end_datetime" in source


def test_local_table_reader_requires_projection_and_chunking():
    signature = inspect.signature(io.read_table)
    assert signature.parameters["columns"].default is inspect.Parameter.empty
    source = inspect.getsource(io.read_table)
    assert "usecols=columns" in source
    assert "chunksize=chunksize" in source
    assert "dataset.to_table(columns=columns, filter=expression)" in source


def test_local_adapter_filters_domains_after_candidate_cohort():
    source = inspect.getsource(base.LocalFileAdapter._load_local_tables)
    assert "_candidate_visit_ids" in source
    assert "allowed_any=allowed_any" in source
