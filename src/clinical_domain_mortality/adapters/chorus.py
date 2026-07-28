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
        from sqlalchemy import bindparam, create_engine, inspect, text

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
            candidate_patients = set(
                raw["encounters"].loc[
                    raw["encounters"][visit_column].isin(candidate_visits),
                    patient_column,
                ]
            )
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
                filters = []
                parameters: dict[str, object] = {}
                statement = text(
                    f"SELECT {select_columns(standard)} FROM {qualified(standard)}"
                )
                source_visit = mapping.get(
                    "visit_id" if standard in {"diagnoses", "bridge"} else "source_visit_id"
                )
                source_patient = mapping.get("patient_id")
                if standard == "diagnoses" and source_patient:
                    filters.append(f"{source_patient} IN :eligible_patients")
                    parameters["eligible_patients"] = sorted(candidate_patients, key=str)
                    statement = statement.bindparams(
                        bindparam("eligible_patients", expanding=True)
                    )
                elif source_visit and source_patient:
                    filters.append(
                        f"({source_visit} IN :eligible_visits OR "
                        f"{source_patient} IN :eligible_patients)"
                    )
                    parameters["eligible_visits"] = sorted(candidate_visits, key=str)
                    parameters["eligible_patients"] = sorted(candidate_patients, key=str)
                    statement = statement.bindparams(
                        bindparam("eligible_visits", expanding=True),
                        bindparam("eligible_patients", expanding=True),
                    )
                elif source_visit:
                    filters.append(f"{source_visit} IN :eligible_visits")
                    parameters["eligible_visits"] = sorted(candidate_visits, key=str)
                    statement = statement.bindparams(
                        bindparam("eligible_visits", expanding=True)
                    )
                elif source_patient:
                    filters.append(f"{source_patient} IN :eligible_patients")
                    parameters["eligible_patients"] = sorted(candidate_patients, key=str)
                    statement = statement.bindparams(
                        bindparam("eligible_patients", expanding=True)
                    )
                event_column = mapping.get(
                    "diagnosis_datetime"
                    if standard == "diagnoses"
                    else "event_datetime"
                )
                if event_column:
                    starts = pd.to_datetime(
                        raw["encounters"].loc[
                            raw["encounters"][visit_column].isin(candidate_visits),
                            encounter_columns["start_datetime"],
                        ]
                    )
                    filters.extend(
                        [f"{event_column} >= :minimum_event_time", f"{event_column} < :maximum_event_time"]
                    )
                    parameters["minimum_event_time"] = starts.min() - pd.Timedelta(
                        days=int(self.config["cohort"]["prior_lookback_days"])
                    )
                    parameters["maximum_event_time"] = starts.max() + pd.Timedelta(
                        hours=float(self.config["cohort"]["predictor_window_hours"])
                    )
                if filters:
                    statement = text(
                        f"SELECT {select_columns(standard)} FROM {qualified(standard)} "
                        f"WHERE {' AND '.join(filters)}"
                    )
                    if "eligible_visits" in parameters:
                        statement = statement.bindparams(
                            bindparam("eligible_visits", expanding=True)
                        )
                    if "eligible_patients" in parameters:
                        statement = statement.bindparams(
                            bindparam("eligible_patients", expanding=True)
                        )
                raw[standard] = pd.read_sql_query(statement, handle, params=parameters)
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
