from __future__ import annotations

import inspect
from copy import deepcopy

import pandas as pd
from sqlalchemy import create_engine

from clinical_domain_mortality import io
from clinical_domain_mortality.adapters import (
    CHoRUSAdapter,
    MIMICIVAdapter,
    base,
    chorus,
)
from clinical_domain_mortality.cohort import build_cohort
from clinical_domain_mortality.config import resolve_project_path
from clinical_domain_mortality.features import prepare_domain_events


def test_chorus_adapter_never_issues_select_star():
    source = inspect.getsource(chorus.CHoRUSAdapter)
    assert "SELECT *" not in source.upper()
    assert "cdm_eligible_acute_cohort" in source
    assert "CREATE TEMPORARY TABLE" in source
    assert "JOIN {relation} AS eligible" in source
    assert "EXISTS (SELECT 1 FROM" in source
    assert "expanding=True" not in source
    assert " IN :eligible" not in source
    assert "predictor_end_datetime" in source


def test_chorus_sql_plan_uses_one_bounded_server_side_cohort_relation(
    chorus_config,
):
    plan = CHoRUSAdapter(chorus_config)._sql_extraction_plan(
        schema_name="fixture_schema",
        dialect="postgresql",
    )
    complete_sql = "\n".join(
        [plan["create_relation"], *plan["queries"].values()]
    ).upper()
    assert "SELECT *" not in complete_sql
    assert "CREATE TEMPORARY TABLE CDM_ELIGIBLE_ACUTE_COHORT" in complete_sql
    assert len(plan["parameters"]) < 25
    assert not any(
        key.startswith(("patient_", "visit_", "encounter_"))
        for key in plan["parameters"]
    )
    for domain in ("measurements", "medications", "procedures"):
        query = plan["queries"][domain].upper()
        assert "CDM_ELIGIBLE_ACUTE_COHORT" in query
        assert "ELIGIBLE.START_DATETIME" in query
        assert "PREDICTOR_END_DATETIME" in query
        assert "IS NOT NULL" in query
        assert "BRIDGE_SRC" in query
        assert " IN (" not in query
    assert "PRIOR_LOOKBACK_DAYS" in plan["queries"]["diagnoses"].upper()
    assert plan["drop_relation"] == (
        "DROP TABLE IF EXISTS cdm_eligible_acute_cohort"
    )


def test_chorus_sql_executes_cohort_first_with_server_attrition(
    tmp_path, monkeypatch, chorus_config
):
    database = tmp_path / "chorus.sqlite"
    engine = create_engine(f"sqlite:///{database}")
    source_root = resolve_project_path(chorus_config["source"]["root"])
    for table_name in chorus_config["source"]["tables"].values():
        pd.read_csv(source_root / f"{table_name}.csv").to_sql(
            table_name,
            engine,
            index=False,
            if_exists="replace",
        )
    config = deepcopy(chorus_config)
    config["source"]["backend"] = "sql"
    config["source"]["sql_dialect"] = "sqlite"
    config["source"]["database_url_env"] = "SYNTHETIC_CHORUS_SQL_URL"
    config["source"].pop("schema_env", None)
    monkeypatch.setenv(
        "SYNTHETIC_CHORUS_SQL_URL",
        f"sqlite:///{database}",
    )
    standardized = CHoRUSAdapter(config).load()
    cohort = build_cohort(standardized, config)
    assert len(cohort.cohort) == 70
    attrition = pd.DataFrame(
        standardized.audit["server_side_cohort_attrition"]
    )
    assert attrition["step"].tolist()[0] == "source encounters"
    assert attrition["step"].tolist()[-1] == (
        "deterministic dataset subsample"
    )
    assert attrition.iloc[0]["visits"] > attrition.iloc[-1]["visits"]
    pd.testing.assert_frame_equal(
        cohort.attrition.reset_index(drop=True),
        attrition.reset_index(drop=True),
        check_dtype=False,
    )
    prepared = prepare_domain_events(standardized, cohort.cohort, config)
    bridge_counts = prepared.audit.loc[
        prepared.audit["status"].eq("linked_bridge"), "count"
    ]
    assert bridge_counts.gt(0).all()


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


def test_native_mimic_large_domains_cannot_bypass_bounded_reader():
    native_source = inspect.getsource(MIMICIVAdapter._load_native)
    reader_source = inspect.getsource(MIMICIVAdapter._native_read)
    assert "pd.read_csv" not in native_source
    assert "pd.read_parquet" not in native_source
    assert "read_table(" in reader_source
    assert "columns=columns" in reader_source
    assert "allowed_any=allowed_any" in reader_source
    assert "time_bounds=time_bounds" in reader_source
    for source_table in ("labevents", "chartevents", "prescriptions"):
        assert source_table in native_source
