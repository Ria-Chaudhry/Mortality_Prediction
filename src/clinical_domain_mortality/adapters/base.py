"""Shared adapter interface and mapping utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import resolve_project_path
from ..errors import ConfigurationError, SchemaError
from ..hashing import hash_file, hash_frame_canonical, hash_object
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
        return self._finalize_standardized(
            tables, {name: len(frame) for name, frame in raw.items()}
        )

    def _finalize_standardized(
        self,
        tables: dict[str, pd.DataFrame],
        source_rows: dict[str, int],
    ) -> StandardizedData:
        """Validate standardized tables and attach value-sensitive provenance."""
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
            "native": self.source.get("native"),
            "release_or_snapshot": self.source.get(
                "release_or_snapshot", self.source.get("expected_version")
            ),
            "deterministic_subsample": self.source.get("deterministic_subsample"),
            "observation_mode": self.source.get("observation_mode"),
            "sql_dialect": self.source.get("sql_dialect"),
            "confirmed": self.source.get("mapping_confirmed"),
        }
        analytical_hashes = {
            name: hash_frame_canonical(frame)
            for name, frame in sorted(tables.items())
            if name != "metadata"
        }
        analytical_hashes["source_release_or_snapshot"] = hash_object(
            self.source.get("release_or_snapshot")
            or self.source.get("expected_version")
            or "synthetic"
        )
        result = StandardizedData(
            tables=tables,
            input_hashes=dict(sorted(analytical_hashes.items())),
            mapping_hash=hash_object(mapping_material),
            audit={
                "adapter": self.config["adapter"],
                "source_rows": source_rows,
                "standardized_rows": {name: len(frame) for name, frame in tables.items()},
                "source_file_digests": dict(sorted(self._input_hashes.items())),
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
        prior_raw = raw.get("prior_encounters")
        tables["prior_encounters"] = (
            self._map(
                prior_raw,
                columns["encounters"],
                STANDARD_COLUMNS["encounters"],
                "encounters",
            )
            if isinstance(prior_raw, pd.DataFrame)
            else tables["encounters"].copy()
        )
        for column in ("age_anchor", "age_anchor_year", "anchor_year_group"):
            if column not in tables["patients"]:
                tables["patients"][column] = pd.NA
        for column in ("race_at_admission", "ethnicity_at_admission"):
            if column not in tables["encounters"]:
                tables["encounters"][column] = pd.NA
            if column not in tables["prior_encounters"]:
                tables["prior_encounters"][column] = pd.NA
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
        exact_death = tables["deaths"]["death_datetime"].notna()
        tables["deaths"]["death_date"] = tables["deaths"]["death_date"].where(
            tables["deaths"]["death_date"].notna(),
            tables["deaths"]["death_datetime"].dt.normalize(),
        )
        tables["deaths"]["death_time_precision"] = tables["deaths"][
            "death_time_precision"
        ].where(
            tables["deaths"]["death_time_precision"].notna(),
            "datetime",
        )
        tables["deaths"]["death_source"] = tables["deaths"]["death_source"].where(
            tables["deaths"]["death_source"].notna(),
            self.source["tables"]["deaths"],
        )
        tables["deaths"]["death_source_conflict"] = tables["deaths"][
            "death_source_conflict"
        ].where(
            tables["deaths"]["death_source_conflict"].notna(),
            False,
        )
        if (~exact_death).any():
            raise SchemaError("Mapped death records require exact death_datetime values")
        for domain in ("measurements", "medications", "procedures"):
            tables[domain]["event_date"] = tables[domain]["event_date"].where(
                tables[domain]["event_date"].notna(),
                tables[domain]["event_datetime"].dt.normalize(),
            )
            tables[domain]["event_time_precision"] = tables[domain][
                "event_time_precision"
            ].where(
                tables[domain]["event_time_precision"].notna(),
                "datetime",
            )
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
            "icd_version",
            "visit_id",
            "source_visit_id",
            "bridge_key",
            "concept_name",
            "value",
            "unit",
            "semantics",
            "source_table",
            "death_date",
            "death_time_precision",
            "death_source",
            "death_source_conflict",
            "event_date",
            "event_time_precision",
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
            "prior_encounters": [
                "start_datetime",
                "end_datetime",
                "followup_end_datetime",
            ],
            "deaths": ["death_datetime", "death_date"],
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
                    "icd_version",
                    "unit",
                    "semantics",
                    "source_table",
                    "bridge_key",
                    "visit_type",
                    "death_time_precision",
                    "death_source",
                    "event_time_precision",
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
        table_paths: dict[str, Any] = {}
        for standard, source_name in self.source["tables"].items():
            if standard in {"bridge", "observations"}:
                try:
                    path = find_table(root, source_name, format_hint)
                except ConfigurationError:
                    raw[standard] = pd.DataFrame()
                    continue
            else:
                path = find_table(root, source_name, format_hint)
            table_paths[standard] = path
            self._input_hashes[path.relative_to(root).as_posix()] = hash_file(path)
        core_order = ("patients", "encounters", "deaths")
        for standard in core_order:
            raw[standard] = read_table(
                table_paths[standard],
                columns=self._source_columns(standard),
            )
        candidate_visits = self._candidate_visit_ids(raw["encounters"])
        encounter_mapping = self.source["columns"]["encounters"]
        visit_source_column = encounter_mapping["visit_id"]
        patient_source_column = encounter_mapping["patient_id"]
        candidate_patients = set(
            raw["encounters"].loc[
                raw["encounters"][visit_source_column].isin(candidate_visits),
                patient_source_column,
            ].tolist()
        )
        start_source_column = encounter_mapping["start_datetime"]
        starts = pd.to_datetime(
            raw["encounters"].loc[
                raw["encounters"][visit_source_column].isin(candidate_visits),
                start_source_column,
            ],
            errors="coerce",
        )
        window = pd.to_timedelta(
            float(self.config["cohort"]["predictor_window_hours"]), unit="h"
        )
        time_range = (
            starts.min(),
            starts.max() + window,
        )
        for standard in ("diagnoses", "measurements", "medications", "procedures"):
            allowed: dict[str, set[Any]] = {}
            allowed_any: dict[str, set[Any]] = {}
            source_visit = self.source["columns"][standard].get(
                "source_visit_id" if standard != "diagnoses" else "visit_id"
            )
            source_patient = self.source["columns"][standard].get("patient_id")
            if standard == "diagnoses" and source_patient:
                allowed[source_patient] = candidate_patients
            else:
                if source_visit:
                    allowed_any[source_visit] = candidate_visits
                if source_patient:
                    allowed_any[source_patient] = candidate_patients
            event_target = (
                "diagnosis_datetime" if standard == "diagnoses" else "event_datetime"
            )
            event_source = self.source["columns"][standard].get(event_target)
            bounds = (
                (
                    event_source,
                    time_range[0]
                    - pd.Timedelta(
                        days=int(self.config["cohort"]["prior_lookback_days"])
                    ),
                    (
                        time_range[1]
                        + pd.Timedelta(
                            days=int(self.config["cohort"]["outcome_horizon_days"])
                        )
                        if standard == "diagnoses"
                        else time_range[1]
                    ),
                )
                if event_source and pd.notna(time_range[0])
                else None
            )
            raw[standard] = read_table(
                table_paths[standard],
                columns=self._source_columns(standard),
                allowed_values=allowed,
                allowed_any=allowed_any,
                time_bounds=bounds,
            )
        for standard in ("bridge", "observations"):
            if standard in raw or standard not in table_paths:
                continue
            if not self.source.get("columns", {}).get(standard):
                raw[standard] = pd.DataFrame()
                continue
            allowed = {}
            visit_column = self.source.get("columns", {}).get(standard, {}).get("visit_id")
            if visit_column:
                allowed[visit_column] = candidate_visits
            raw[standard] = read_table(
                table_paths[standard],
                columns=self._source_columns(standard),
                allowed_values=allowed,
            )
        return raw

    def _source_columns(self, standard: str) -> list[str]:
        mapping = self.source.get("columns", {}).get(standard, {})
        columns = [str(value) for value in mapping.values() if value]
        if not columns:
            raise ConfigurationError(f"No source columns configured for {standard}")
        return list(dict.fromkeys(columns))

    def _candidate_visit_ids(self, encounters: pd.DataFrame) -> set[Any]:
        """Apply source-mapped acute/non-elective predicates before event scans."""
        mapping = self.source["columns"]["encounters"]
        visit_column = mapping["visit_id"]
        keep = pd.Series(True, index=encounters.index)
        visit_type = mapping.get("visit_type")
        if visit_type:
            acute = {
                str(value).casefold()
                for value in self.config["cohort"]["acute_visit_types"]
            }
            keep &= encounters[visit_type].astype("string").str.casefold().isin(acute)
        elective = mapping.get("elective")
        if elective:
            excluded = {
                str(value).strip().casefold()
                for value in self.config["cohort"]["excluded_elective_values"]
            }
            keep &= ~encounters[elective].astype("string").str.strip().str.casefold().isin(
                excluded
            )
        return set(encounters.loc[keep, visit_column].tolist())
