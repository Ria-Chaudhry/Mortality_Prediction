"""Configurable OMOP-compatible CHoRUS adapter."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ..config import require_environment_reference
from ..errors import ConfigurationError
from ..hashing import hash_object
from .base import LocalFileAdapter, StandardizedData


class CHoRUSAdapter(LocalFileAdapter):
    """Normalize configured CHoRUS/OMOP tables without embedding site details."""

    def load(self) -> StandardizedData:
        backend = self.source.get("backend", "files")
        if backend == "files":
            self._server_attrition = pd.DataFrame()
            raw = self._load_local_tables()
        elif backend == "sql":
            raw = self._load_sql_tables()
        else:
            raise ConfigurationError(f"Unsupported CHoRUS backend: {backend}")
        observations = raw.pop("observations", pd.DataFrame())
        result = self._build_result(raw)
        server_attrition = getattr(self, "_server_attrition", pd.DataFrame())
        if not server_attrition.empty:
            result.audit["server_side_cohort_attrition"] = (
                server_attrition[["step", "visits", "patients"]]
                .to_dict(orient="records")
            )
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

        plan = self._sql_extraction_plan(
            schema_name=schema_name,
            dialect=str(self.source.get("sql_dialect", "postgresql")),
        )
        self._input_hashes["sql_extraction_plan"] = hash_object(
            {
                "create_relation": plan["create_relation"],
                "queries": plan["queries"],
                "drop_relation": plan["drop_relation"],
                "parameters": plan["parameters"],
                "dialect": plan["dialect"],
            }
        )
        with engine.begin() as handle:
            handle.execute(text(plan["create_relation"]), plan["parameters"])
            try:
                for standard, query in plan["queries"].items():
                    if standard in raw:
                        continue
                    raw[standard] = pd.read_sql_query(
                        text(query),
                        handle,
                        params=plan["parameters"],
                    )
            finally:
                handle.execute(text(plan["drop_relation"]))
        self._server_attrition = raw.pop(
            "_cohort_attrition",
            pd.DataFrame(columns=["step", "visits", "patients"]),
        )
        self._input_hashes["sql_extraction_signature"] = self._sql_signature(raw)
        return raw

    def _sql_extraction_plan(
        self,
        *,
        schema_name: str | None,
        dialect: str,
    ) -> dict[str, Any]:
        """Build a cohort-first SQL plan without cohort-sized parameters.

        The plan is inspectable without a database. Only PostgreSQL and SQLite
        expressions are emitted; adding a backend requires an explicit,
        regression-tested capability branch.
        """
        if dialect not in {"postgresql", "sqlite"}:
            raise ConfigurationError(
                f"Unsupported CHoRUS SQL dialect capability: {dialect!r}"
            )

        def qualified(standard: str) -> str:
            table_name = self.source["tables"][standard]
            return f"{schema_name}.{table_name}" if schema_name else table_name

        def projected(standard: str, alias: str = "src") -> str:
            return ", ".join(
                f"{alias}.{name}" for name in self._source_columns(standard)
            )

        def stage_projected(alias: str = "s") -> str:
            columns = [
                *(f"{alias}.{name}" for name in self._source_columns("encounters")),
                f"{alias}._birth_datetime",
                f"{alias}._death_datetime",
            ]
            return ", ".join(columns)

        encounter = self.source["columns"]["encounters"]
        patient = self.source["columns"]["patients"]
        death = self.source["columns"]["deaths"]
        visit_id = encounter["visit_id"]
        encounter_patient = encounter["patient_id"]
        start = encounter["start_datetime"]
        end = encounter["end_datetime"]
        visit_type = encounter["visit_type"]
        elective = encounter["elective"]
        followup = encounter["followup_end_datetime"]
        birth = patient["birth_datetime"]
        patient_id = patient["patient_id"]
        death_patient = death["patient_id"]
        death_time = death["death_datetime"]
        relation = "cdm_eligible_acute_cohort"
        if self.source.get("deterministic_subsample", {}).get("enabled"):
            raise ConfigurationError(
                "CHoRUS SQL cohort relation does not support source subsampling"
            )

        if dialect == "postgresql":
            stage_predictor_end = (
                f"s.{start} + "
                "(:predictor_window_hours * INTERVAL '1 hour')"
            )
            stage_age = (
                f"EXTRACT(EPOCH FROM (s.{start} - s._birth_datetime)) "
                "/ (365.2425 * 86400)"
            )
            stage_landmark = (
                f"s.{start} + (:landmark_hours * INTERVAL '1 hour')"
            )
            stage_horizon = (
                f"s.{start} + (:outcome_horizon_days * INTERVAL '1 day')"
            )
            prior_lower = (
                "eligible.start_datetime - "
                "(:prior_lookback_days * INTERVAL '1 day')"
            )
        else:
            stage_predictor_end = (
                f"datetime(s.{start}, '+' || "
                ":predictor_window_hours || ' hours')"
            )
            stage_age = (
                f"(julianday(s.{start}) - julianday(s._birth_datetime)) "
                "/ 365.2425"
            )
            stage_landmark = (
                f"datetime(s.{start}, '+' || :landmark_hours || ' hours')"
            )
            stage_horizon = (
                f"datetime(s.{start}, '+' || :outcome_horizon_days || ' days')"
            )
            prior_lower = (
                "datetime(eligible.start_datetime, '-' || "
                ":prior_lookback_days || ' days')"
            )

        acute = [str(value).casefold() for value in self.config["cohort"]["acute_visit_types"]]
        excluded = [
            str(value).strip().casefold()
            for value in self.config["cohort"]["excluded_elective_values"]
        ]
        parameters: dict[str, Any] = {
            "predictor_window_hours": float(
                self.config["cohort"]["predictor_window_hours"]
            ),
            "landmark_hours": float(self.config["cohort"]["landmark_hours"]),
            "outcome_horizon_days": int(
                self.config["cohort"]["outcome_horizon_days"]
            ),
            "prior_lookback_days": int(
                self.config["cohort"]["prior_lookback_days"]
            ),
            "min_age_years": float(self.config["cohort"]["min_age_years"]),
            "max_age_years": float(self.config["cohort"]["max_age_years"]),
        }
        acute_parameters = []
        for index, value in enumerate(acute):
            key = f"acute_type_{index}"
            parameters[key] = value
            acute_parameters.append(f":{key}")
        elective_parameters = []
        for index, value in enumerate(excluded):
            key = f"excluded_elective_{index}"
            parameters[key] = value
            elective_parameters.append(f":{key}")

        short_predicate = (
            f"WHERE s.{end} IS NULL OR s.{end} >= {stage_landmark}"
            if not self.config["cohort"].get("retain_short_visits")
            else ""
        )
        verified_predicate = (
            f"WHERE s._death_datetime IS NOT NULL "
            f"OR s.{followup} >= {stage_horizon}"
            if self.config["cohort"].get("require_verified_followup")
            else ""
        )
        ctes = f"""
death_by_patient AS (
    SELECT
        CAST({death_patient} AS VARCHAR) AS _death_patient_id,
        MIN({death_time}) AS _death_datetime
    FROM {qualified("deaths")}
    GROUP BY CAST({death_patient} AS VARCHAR)
),
source_stage AS (
    SELECT
        {projected("encounters", "v")},
        p.{birth} AS _birth_datetime,
        d._death_datetime
    FROM {qualified("encounters")} AS v
    LEFT JOIN {qualified("patients")} AS p
      ON CAST(p.{patient_id} AS VARCHAR) =
         CAST(v.{encounter_patient} AS VARCHAR)
    LEFT JOIN death_by_patient AS d
      ON d._death_patient_id = CAST(v.{encounter_patient} AS VARCHAR)
),
adult_stage AS (
    SELECT {stage_projected()} FROM source_stage AS s
    WHERE {stage_age} BETWEEN :min_age_years AND :max_age_years
),
acute_stage AS (
    SELECT {stage_projected()} FROM adult_stage AS s
    WHERE LOWER(CAST(s.{visit_type} AS VARCHAR))
      IN ({", ".join(acute_parameters)})
),
non_elective_stage AS (
    SELECT {stage_projected()} FROM acute_stage AS s
    WHERE COALESCE(LOWER(CAST(s.{elective} AS VARCHAR)), '')
      NOT IN ({", ".join(elective_parameters)})
),
short_stage AS (
    SELECT {stage_projected()} FROM non_elective_stage AS s
    {short_predicate}
),
landmark_stage AS (
    SELECT {stage_projected()} FROM short_stage AS s
    WHERE s._death_datetime IS NULL
       OR s._death_datetime > {stage_landmark}
),
verified_stage AS (
    SELECT {stage_projected()} FROM landmark_stage AS s
    {verified_predicate}
)
""".strip()
        create_relation = f"""
CREATE TEMPORARY TABLE {relation} AS
WITH {ctes}
SELECT
    CAST(s.{visit_id} AS VARCHAR) AS visit_id,
    CAST(s.{encounter_patient} AS VARCHAR) AS patient_id,
    s.{start} AS start_datetime,
    CASE
      WHEN s.{end} IS NOT NULL AND s.{end} < {stage_predictor_end}
        THEN s.{end}
      ELSE {stage_predictor_end}
    END AS predictor_end_datetime
FROM verified_stage AS s
""".strip()

        def attrition_count(
            order: int, step: str, stage: str
        ) -> str:
            return (
                f"SELECT {order} AS stage_order, '{step}' AS step, "
                f"COUNT(*) AS visits, "
                f"COUNT(DISTINCT CAST({encounter_patient} AS VARCHAR)) "
                f"AS patients FROM {stage}"
            )

        attrition_query = (
            f"WITH {ctes}\n"
            + "\nUNION ALL\n".join(
                [
                    attrition_count(1, "source encounters", "source_stage"),
                    attrition_count(2, "adult age range", "adult_stage"),
                    attrition_count(3, "acute encounter type", "acute_stage"),
                    attrition_count(4, "non-elective", "non_elective_stage"),
                    attrition_count(5, "short-visit policy", "short_stage"),
                    attrition_count(
                        6,
                        "alive after 24-hour landmark",
                        "landmark_stage",
                    ),
                    attrition_count(
                        7,
                        "verified 30-day follow-up",
                        "verified_stage",
                    ),
                    attrition_count(
                        8,
                        "deterministic dataset subsample",
                        "verified_stage",
                    ),
                ]
            )
            + "\nORDER BY stage_order"
        )

        patient_match = (
            f"CAST(src.{patient_id} AS VARCHAR) = eligible.patient_id"
        )
        encounter_match = (
            f"CAST(src.{encounter_patient} AS VARCHAR) = eligible.patient_id"
        )
        queries: dict[str, str] = {
            "_cohort_attrition": attrition_query,
            "patients": (
                f"SELECT {projected('patients')} FROM {qualified('patients')} AS src "
                f"WHERE EXISTS (SELECT 1 FROM {relation} AS eligible "
                f"WHERE {patient_match}) ORDER BY src.{patient_id}"
            ),
            "encounters": (
                f"SELECT {projected('encounters')} FROM {qualified('encounters')} AS src "
                f"JOIN {relation} AS eligible ON "
                f"CAST(src.{visit_id} AS VARCHAR) = eligible.visit_id "
                f"ORDER BY src.{start}, src.{encounter_patient}, src.{visit_id}"
            ),
            "prior_encounters": (
                f"SELECT DISTINCT {projected('encounters')} "
                f"FROM {qualified('encounters')} AS src "
                f"JOIN {relation} AS eligible ON {encounter_match} "
                f"AND src.{start} < eligible.start_datetime "
                f"AND src.{start} >= {prior_lower} "
                f"ORDER BY src.{start}, src.{encounter_patient}, src.{visit_id}"
            ),
            "deaths": (
                f"SELECT {projected('deaths')} FROM {qualified('deaths')} AS src "
                f"WHERE EXISTS (SELECT 1 FROM {relation} AS eligible WHERE "
                f"CAST(src.{death_patient} AS VARCHAR) = eligible.patient_id) "
                f"ORDER BY src.{death_patient}, src.{death_time}"
            ),
        }

        for standard in (
            "diagnoses",
            "measurements",
            "medications",
            "procedures",
            "bridge",
            "observations",
        ):
            mapping = self.source.get("columns", {}).get(standard, {})
            if not mapping:
                continue
            source_visit = mapping.get(
                "visit_id" if standard in {"diagnoses", "bridge"} else "source_visit_id"
            )
            source_patient = mapping.get("patient_id")
            source_bridge = mapping.get("bridge_key")
            event_column = mapping.get(
                "diagnosis_datetime" if standard == "diagnoses" else "event_datetime"
            )
            visit_match = (
                f"CAST(src.{source_visit} AS VARCHAR) = eligible.visit_id"
                if source_visit
                else "FALSE"
            )
            source_patient_match = (
                f"CAST(src.{source_patient} AS VARCHAR) = eligible.patient_id"
                if source_patient
                else "FALSE"
            )
            if standard == "diagnoses":
                if not source_patient or not event_column:
                    raise ConfigurationError(
                        "CHoRUS SQL diagnoses require patient and diagnosis-time mappings"
                    )
                relation_filter = (
                    f"{source_patient_match} AND src.{event_column} >= {prior_lower} "
                    f"AND src.{event_column} < eligible.start_datetime"
                )
            elif standard == "bridge":
                relation_filter = visit_match
            elif event_column:
                # Explicit visit IDs are authoritative. Patient-time fallback is
                # admitted by SQL only when the source visit field is null.
                identity_parts = []
                if source_visit:
                    identity_parts.append(
                        f"(src.{source_visit} IS NOT NULL AND {visit_match})"
                    )
                if (
                    source_bridge
                    and self.source.get("columns", {}).get("bridge")
                ):
                    bridge_mapping = self.source["columns"]["bridge"]
                    bridge_key = bridge_mapping["bridge_key"]
                    bridge_visit = bridge_mapping["visit_id"]
                    no_explicit = (
                        f"src.{source_visit} IS NULL AND "
                        if source_visit
                        else ""
                    )
                    identity_parts.append(
                        f"({no_explicit}src.{source_bridge} IS NOT NULL AND "
                        f"EXISTS (SELECT 1 FROM {qualified('bridge')} AS bridge_src "
                        f"WHERE CAST(bridge_src.{bridge_key} AS VARCHAR) = "
                        f"CAST(src.{source_bridge} AS VARCHAR) AND "
                        f"CAST(bridge_src.{bridge_visit} AS VARCHAR) = "
                        "eligible.visit_id))"
                    )
                no_explicit_or_bridge = " AND ".join(
                    [
                        *(
                            [f"src.{source_visit} IS NULL"]
                            if source_visit
                            else []
                        ),
                        *(
                            [f"src.{source_bridge} IS NULL"]
                            if source_bridge
                            else []
                        ),
                    ]
                )
                if source_patient:
                    identity_parts.append(
                        f"({no_explicit_or_bridge} AND {source_patient_match})"
                        if no_explicit_or_bridge
                        else source_patient_match
                    )
                if not identity_parts:
                    raise ConfigurationError(
                        f"CHoRUS SQL {standard} mapping lacks a supported "
                        "visit, bridge, or patient linkage field"
                    )
                identity_filter = f"({' OR '.join(identity_parts)})"
                relation_filter = (
                    f"{identity_filter} AND "
                    f"src.{event_column} >= eligible.start_datetime AND "
                    f"src.{event_column} < eligible.predictor_end_datetime"
                )
            else:
                relation_filter = visit_match
            queries[standard] = (
                f"SELECT {projected(standard)} FROM {qualified(standard)} AS src "
                f"WHERE EXISTS (SELECT 1 FROM {relation} AS eligible "
                f"WHERE {relation_filter})"
            )

        return {
            "create_relation": create_relation,
            "queries": queries,
            "drop_relation": f"DROP TABLE IF EXISTS {relation}",
            "parameters": parameters,
            "dialect": dialect,
        }

    @staticmethod
    def _sql_signature(raw: dict[str, pd.DataFrame]) -> str:
        from ..hashing import hash_frame_canonical

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
