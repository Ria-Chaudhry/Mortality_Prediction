"""Configurable OMOP-compatible CHoRUS adapter."""

from __future__ import annotations

import re

import pandas as pd

from ..config import require_environment_reference
from ..errors import ConfigurationError
from .base import LocalFileAdapter, StandardizedData


class CHoRUSAdapter(LocalFileAdapter):
    """Normalize configured CHoRUS/OMOP tables without embedding site details."""

    def load(self) -> StandardizedData:
        backend = self.source.get("backend", "files")
        if backend == "files":
            raw = self._load_local_tables()
        elif backend == "sql":
            raw = self._load_sql_tables()
        else:
            raise ConfigurationError(f"Unsupported CHoRUS backend: {backend}")
        observations = raw.pop("observations", pd.DataFrame())
        result = self._build_result(raw)
        result.audit["observation_rows"] = len(observations)
        result.audit["observation_mode"] = self.source.get("observation_mode", "audit_only")
        if not observations.empty and self.source.get("observation_mode") == "numeric_measurements":
            mapping = self.source.get("columns", {}).get("observations")
            if not mapping:
                raise ConfigurationError(
                    "numeric_measurements observation mode requires an explicit observation mapping"
                )
            from ..schemas import EVENT_COLUMNS

            normalized = self._map(observations, mapping, EVENT_COLUMNS, "observations")
            normalized["event_datetime"] = pd.to_datetime(
                normalized["event_datetime"], errors="coerce"
            )
            if normalized["event_datetime"].isna().any():
                raise ConfigurationError("Mapped CHoRUS observations contain invalid event times")
            self._normalize_strings({"observations": normalized})
            result.tables["measurements"] = pd.concat(
                [result.tables["measurements"], normalized], ignore_index=True
            )
            result.tables["metadata"] = pd.concat(
                [
                    result.tables["metadata"],
                    normalized.assign(domain="measurements")[
                        [
                            "domain",
                            "concept_key",
                            "concept_name",
                            "source_table",
                            "semantics",
                            "unit",
                        ]
                    ],
                ],
                ignore_index=True,
            ).drop_duplicates()
            self.validate(result)
        return result

    def _load_sql_tables(self) -> dict[str, pd.DataFrame]:
        from sqlalchemy import (
            Column,
            DateTime,
            MetaData,
            String,
            Table,
            create_engine,
            inspect,
            text,
        )

        connection = require_environment_reference(self.source["database_url_env"])
        schema_name = None
        schema_env = self.source.get("schema_env")
        if schema_env:
            schema_name = require_environment_reference(schema_env)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema_name):
                raise ConfigurationError("Unsafe SQL schema name")
        engine = create_engine(connection)
        inspector = inspect(engine)
        raw: dict[str, pd.DataFrame] = {}
        optional = {"bridge", "observations"}
        for standard, table_name in self.source["tables"].items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
                raise ConfigurationError(f"Unsafe SQL table name: {table_name!r}")
            if standard in optional and not self.source.get("columns", {}).get(standard):
                raw[standard] = pd.DataFrame()
                continue
            for column in self._source_columns(standard):
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column):
                    raise ConfigurationError(f"Unsafe SQL column name: {column!r}")
            if not inspector.has_table(table_name, schema=schema_name):
                if standard in optional:
                    raw[standard] = pd.DataFrame()
                    continue
                raise ConfigurationError(
                    f"Configured CHoRUS table does not exist: {table_name}"
                )

        def select_columns(standard: str) -> str:
            return ", ".join(self._source_columns(standard))

        def qualified(standard: str) -> str:
            table_name = self.source["tables"][standard]
            return f"{schema_name}.{table_name}" if schema_name else table_name

        with engine.connect() as handle:
            for standard in ("patients", "encounters", "deaths"):
                raw[standard] = pd.read_sql_query(
                    text(
                        f"SELECT {select_columns(standard)} FROM {qualified(standard)}"
                    ),
                    handle,
                )
            candidate_visits = self._candidate_visit_ids(raw["encounters"])
            encounter_columns = self.source["columns"]["encounters"]
            patient_column = encounter_columns["patient_id"]
            visit_column = encounter_columns["visit_id"]
            candidate_frame = raw["encounters"].loc[
                raw["encounters"][visit_column].isin(candidate_visits),
                [
                    visit_column,
                    patient_column,
                    encounter_columns["start_datetime"],
                ],
            ].copy()
            candidate_frame.columns = [
                "visit_id",
                "patient_id",
                "start_datetime",
            ]
            candidate_frame["visit_id"] = candidate_frame["visit_id"].astype(str)
            candidate_frame["patient_id"] = candidate_frame["patient_id"].astype(str)
            candidate_frame["start_datetime"] = pd.to_datetime(
                candidate_frame["start_datetime"], errors="raise"
            )
            candidate_frame["predictor_end_datetime"] = candidate_frame[
                "start_datetime"
            ] + pd.to_timedelta(
                float(self.config["cohort"]["predictor_window_hours"]), unit="h"
            )
            staging_name = "cdm_candidate_acute_cohort"
            metadata = MetaData()
            staging = Table(
                staging_name,
                metadata,
                Column("visit_id", String, primary_key=True),
                Column("patient_id", String, nullable=False, index=True),
                Column("start_datetime", DateTime, nullable=False),
                Column("predictor_end_datetime", DateTime, nullable=False),
                prefixes=["TEMPORARY"],
            )
            staging.create(handle)
            records = candidate_frame.to_dict(orient="records")
            chunk_size = int(self.source.get("staging_insert_chunk_size", 1000))
            for offset in range(0, len(records), chunk_size):
                handle.execute(staging.insert(), records[offset : offset + chunk_size])
            for standard in (
                "diagnoses",
                "measurements",
                "medications",
                "procedures",
                "bridge",
                "observations",
            ):
                if standard in raw:
                    continue
                mapping = self.source.get("columns", {}).get(standard, {})
                if not mapping:
                    raw[standard] = pd.DataFrame()
                    continue
                filters: list[str] = []
                source_visit = mapping.get(
                    "visit_id" if standard in {"diagnoses", "bridge"} else "source_visit_id"
                )
                source_patient = mapping.get("patient_id")
                event_column = mapping.get(
                    "diagnosis_datetime"
                    if standard == "diagnoses"
                    else "event_datetime"
                )
                visit_match = (
                    f"CAST(src.{source_visit} AS VARCHAR) = eligible.visit_id"
                    if source_visit
                    else "FALSE"
                )
                patient_match = (
                    f"CAST(src.{source_patient} AS VARCHAR) = eligible.patient_id"
                    if source_patient
                    else "FALSE"
                )
                if standard == "diagnoses":
                    if not source_patient or not event_column:
                        raise ConfigurationError(
                            "CHoRUS SQL diagnoses require patient and diagnosis-time mappings"
                        )
                    lookback = int(self.config["cohort"]["prior_lookback_days"])
                    relation_filter = (
                        f"{patient_match} AND "
                        f"src.{event_column} >= eligible.start_datetime "
                        f"- INTERVAL '{lookback} days' AND "
                        f"src.{event_column} < eligible.start_datetime"
                    )
                elif standard == "bridge":
                    relation_filter = visit_match
                elif event_column:
                    relation_filter = (
                        f"({visit_match} OR {patient_match}) AND "
                        f"src.{event_column} >= eligible.start_datetime AND "
                        f"src.{event_column} < eligible.predictor_end_datetime"
                    )
                else:
                    relation_filter = f"({visit_match} OR {patient_match})"
                filters.append(
                    "EXISTS (SELECT 1 FROM "
                    f"{staging_name} AS eligible WHERE {relation_filter})"
                )
                statement = text(
                    f"SELECT {', '.join(f'src.{name}' for name in self._source_columns(standard))} "
                    f"FROM {qualified(standard)} AS src "
                    f"WHERE {' AND '.join(filters)}"
                )
                raw[standard] = pd.read_sql_query(statement, handle)
        self._input_hashes["sql_extraction_signature"] = self._sql_signature(raw)
        return raw

    @staticmethod
    def _sql_signature(raw: dict[str, pd.DataFrame]) -> str:
        from ..hashing import hash_frame_canonical, hash_object

        return hash_object(
            {
                name: {
                    "columns": sorted(frame.columns),
                    "rows": len(frame),
                    "content": hash_frame_canonical(frame),
                }
                for name, frame in sorted(raw.items())
            }
        )
