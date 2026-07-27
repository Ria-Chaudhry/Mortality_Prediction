"""Shared adapter interface and mapping utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import resolve_project_path
from ..errors import ConfigurationError, SchemaError
from ..hashing import hash_file, hash_object
from ..io import find_table, read_table
from ..schemas import EVENT_COLUMNS, STANDARD_COLUMNS, validate_standardized


@dataclass
class StandardizedData:
    """Source-neutral clinical tables plus provenance."""

    tables: dict[str, pd.DataFrame]
    input_hashes: dict[str, str]
    mapping_hash: str
    audit: dict[str, Any] = field(default_factory=dict)


class SourceAdapter(ABC):
    """Contract implemented by CHoRUS and MIMIC-IV adapters."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.source = config["source"]
        self._input_hashes: dict[str, str] = {}

    @abstractmethod
    def load(self) -> StandardizedData:
        """Read and normalize all required clinical tables."""

    def validate(self, result: StandardizedData) -> None:
        validate_standardized(result.tables)

    def _build_result(self, raw: dict[str, pd.DataFrame]) -> StandardizedData:
        tables = self._normalize(raw)
        metadata_parts = []
        for domain in ("measurements", "medications", "procedures"):
            event = tables[domain]
            if event.empty:
                continue
            part = (
                event.assign(domain=domain)[
                    ["domain", "concept_key", "concept_name", "source_table", "semantics", "unit"]
                ]
                .drop_duplicates()
                .sort_values(["domain", "concept_key", "source_table", "semantics"], kind="stable")
            )
            metadata_parts.append(part)
        tables["metadata"] = (
            pd.concat(metadata_parts, ignore_index=True)
            if metadata_parts
            else pd.DataFrame(
                columns=["domain", "concept_key", "concept_name", "source_table", "semantics", "unit"]
            )
        )
        mapping_material = {
            "tables": self.source.get("tables"),
            "columns": self.source.get("columns"),
            "semantics": self.source.get("source_semantics"),
            "observation_mode": self.source.get("observation_mode"),
            "confirmed": self.source.get("mapping_confirmed"),
        }
        result = StandardizedData(
            tables=tables,
            input_hashes=dict(sorted(self._input_hashes.items())),
            mapping_hash=hash_object(mapping_material),
            audit={
                "adapter": self.config["adapter"],
                "source_rows": {name: len(frame) for name, frame in raw.items()},
                "standardized_rows": {name: len(frame) for name, frame in tables.items()},
            },
        )
        self.validate(result)
        return result

    def _normalize(self, raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        columns = self.source["columns"]
        tables: dict[str, pd.DataFrame] = {}
        for name in ("patients", "encounters", "deaths", "diagnoses"):
            required = STANDARD_COLUMNS[name]
            tables[name] = self._map(raw[name], columns[name], required, name)
        for domain in ("measurements", "medications", "procedures"):
            tables[domain] = self._map(raw[domain], columns[domain], EVENT_COLUMNS, domain)
            tables[domain]["source_table"] = self.source["tables"][domain]
        bridge_raw = raw.get("bridge", pd.DataFrame())
        if bridge_raw.empty:
            tables["bridge"] = pd.DataFrame(columns=STANDARD_COLUMNS["bridge"])
        else:
            tables["bridge"] = self._map(
                bridge_raw, columns["bridge"], STANDARD_COLUMNS["bridge"], "bridge"
            )
        self._parse_dates(tables)
        self._normalize_strings(tables)
        tables["deaths"] = (
            tables["deaths"]
            .sort_values(["patient_id", "death_datetime"], kind="stable")
            .drop_duplicates(["patient_id"], keep="first")
            .reset_index(drop=True)
        )
        return tables

    def _map(
        self,
        frame: pd.DataFrame,
        mapping: dict[str, str],
        targets: list[str],
        table_name: str,
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        optional = {
            "diagnosis_id",
            "visit_id",
            "source_visit_id",
            "bridge_key",
            "concept_name",
            "value",
            "unit",
            "semantics",
            "source_table",
        }
        for target in targets:
            source_column = mapping.get(target)
            if source_column and source_column in frame.columns:
                result[target] = frame[source_column]
            elif target in optional:
                if target == "diagnosis_id":
                    result[target] = [f"{table_name}:{index}" for index in range(len(frame))]
                else:
                    result[target] = pd.NA
            else:
                raise SchemaError(
                    f"{table_name} mapping for {target!r} is absent or source column is missing"
                )
        if "source_table" in targets:
            result["source_table"] = self.source["tables"][table_name]
        if "semantics" in targets:
            default = self.source.get("source_semantics", {}).get(table_name, {}).get(
                self.source["tables"][table_name]
            )
            if default and result["semantics"].isna().any():
                result["semantics"] = result["semantics"].fillna(default)
            if result["semantics"].isna().any():
                raise ConfigurationError(f"{table_name} includes records without explicit semantics")
        return result.reset_index(drop=True)

    @staticmethod
    def _parse_dates(tables: dict[str, pd.DataFrame]) -> None:
        date_columns = {
            "patients": ["birth_datetime"],
            "encounters": ["start_datetime", "end_datetime", "followup_end_datetime"],
            "deaths": ["death_datetime"],
            "diagnoses": ["diagnosis_datetime"],
            "measurements": ["event_datetime"],
            "medications": ["event_datetime"],
            "procedures": ["event_datetime"],
        }
        for name, columns in date_columns.items():
            for column in columns:
                tables[name][column] = pd.to_datetime(tables[name][column], errors="coerce")
            if name not in {"deaths"} and tables[name][columns[0]].isna().any():
                raise SchemaError(f"{name}.{columns[0]} contains missing or invalid datetimes")

    @staticmethod
    def _normalize_strings(tables: dict[str, pd.DataFrame]) -> None:
        for _name, frame in tables.items():
            for column in frame.columns:
                if column.endswith("_id") or column in {
                    "concept_key",
                    "concept_name",
                    "code",
                    "unit",
                    "semantics",
                    "source_table",
                    "bridge_key",
                    "visit_type",
                }:
                    frame[column] = frame[column].astype("string")


class LocalFileAdapter(SourceAdapter):
    """Common local CSV/compressed-CSV/Parquet table loading."""

    def _load_local_tables(self) -> dict[str, pd.DataFrame]:
        root = resolve_project_path(self.source["root"])
        if not root.is_dir():
            raise ConfigurationError(f"Source root does not exist: {root}")
        raw: dict[str, pd.DataFrame] = {}
        format_hint = self.source.get("file_format", "auto")
        for standard, source_name in self.source["tables"].items():
            if standard in {"bridge", "observations"}:
                try:
                    path = find_table(root, source_name, format_hint)
                except ConfigurationError:
                    raw[standard] = pd.DataFrame()
                    continue
            else:
                path = find_table(root, source_name, format_hint)
            raw[standard] = read_table(path)
            self._input_hashes[path.relative_to(root).as_posix()] = hash_file(path)
        return raw
