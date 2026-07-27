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
        from sqlalchemy import create_engine, inspect, text

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
        with engine.connect() as handle:
            for standard, table_name in self.source["tables"].items():
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
                    raise ConfigurationError(f"Unsafe SQL table name: {table_name!r}")
                if not inspector.has_table(table_name, schema=schema_name):
                    if standard in {"bridge", "observations"}:
                        raw[standard] = pd.DataFrame()
                        continue
                    raise ConfigurationError(f"Configured CHoRUS table does not exist: {table_name}")
                qualified = f"{schema_name}.{table_name}" if schema_name else table_name
                raw[standard] = pd.read_sql_query(text(f"SELECT * FROM {qualified}"), handle)
        self._input_hashes["sql_source_signature"] = self._sql_signature(raw)
        return raw

    @staticmethod
    def _sql_signature(raw: dict[str, pd.DataFrame]) -> str:
        from ..hashing import hash_object

        return hash_object(
            {
                name: {"columns": sorted(frame.columns), "rows": len(frame)}
                for name, frame in sorted(raw.items())
            }
        )
