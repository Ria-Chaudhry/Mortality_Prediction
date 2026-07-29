"""End-to-end source-neutral execution and release verification."""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import pyarrow
import scipy
import shap
import sklearn

from .adapters import CHoRUSAdapter, MIMICIVAdapter, SourceAdapter
from .audit import git_commit, git_is_dirty, scan_public_tree, utc_timestamp
from .cohort import build_cohort, create_patient_folds
from .config import PROJECT_ROOT, load_config, read_yaml, resolve_project_path
from .errors import ConfigurationError, CountMismatchError, IntegrityError, LinkageError
from .evaluation import evaluate_predictions
from .features import (
    assemble_matrix,
    build_fold_domain_features,
    prepare_domain_events,
    select_concepts,
)
from .hashing import hash_file, hash_frame_schema, hash_frame_values, hash_object
from .io import read_json, verify_hashes, write_csv, write_json
from .modeling import (
    fit_predict_fold,
    fold_shap_aggregate,
    validate_oof_predictions,
)


@dataclass
class RunResult:
    dataset: str
    public_dir: Path
    restricted_dir: Path
    run_manifest: dict[str, Any]


def adapter_for(config: dict[str, Any]) -> SourceAdapter:
    if config["adapter"] == "chorus":
        return CHoRUSAdapter(config)
    if config["adapter"] == "mimiciv":
        return MIMICIVAdapter(config)
    raise ConfigurationError(f"Unsupported adapter: {config['adapter']}")


def paper_preflight(config_path: str | Path) -> dict[str, Any]:
    """Validate paper-mode facts without opening a source or creating outputs."""
    try:
        config = load_config(config_path)
    except ConfigurationError as error:
        return {
            "status": "blocked",
            "source_access_attempted": False,
            "reason": str(error),
        }
    if not config.get("paper_run"):
        raise ConfigurationError("Paper preflight requires paper_run: true")
    return {
        "status": "ready",
        "dataset": config["dataset"],
        "config_hash": config["_meta"]["config_hash"],
        "source_access_attempted": False,
    }


def run_pipeline(
    config_path: str | Path,
    output_base: str | Path | None = None,
    restricted_base: str | Path | None = None,
    stop_after: int = 8,
) -> RunResult:
    """Run stages 1-8, optionally stopping after an auditable intermediate stage."""
    config = load_config(config_path)
    dataset = str(config["dataset"])
    public_base = (
        Path(output_base).resolve()
        if output_base
        else resolve_project_path(config["outputs"]["public_root"])
    )
    private_base = (
        Path(restricted_base).resolve()
        if restricted_base
        else resolve_project_path(config["outputs"]["restricted_root"])
    )
    restricted_dir = private_base / dataset
    public_dir = (
        public_base / dataset
        if config.get("synthetic")
        else restricted_dir / "release_candidate_aggregate"
    )
    if restricted_dir.exists():
        shutil.rmtree(restricted_dir)
    if public_dir.exists():
        shutil.rmtree(public_dir)
    public_dir.mkdir(parents=True)
    restricted_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "manifests").mkdir()

    # Stage 1: validate and normalize source.
    standardized = adapter_for(config).load()
    write_json(
        {
            "mapping_confirmed": True,
            "mapping_hash": standardized.mapping_hash,
            "adapter": config["adapter"],
            "table_schema_hashes": {
                name: hash_frame_schema(frame)
                for name, frame in standardized.tables.items()
            },
            "table_row_counts": {
                name: len(frame) for name, frame in standardized.tables.items()
            },
        },
        public_dir / "source_mapping_validation.json",
    )
    write_json(
        {
            "mapping_confirmed": True,
            "mapping_hash": standardized.mapping_hash,
            "adapter": config["adapter"],
            "input_table_count": len(standardized.input_hashes),
        },
        public_dir / "manifests" / "mapping_manifest.json",
    )
    if stop_after == 1:
        return _partial_result(config, public_dir, restricted_dir, standardized)

    # Stage 2: freeze cohort, outcome, baseline, and row order.
    try:
        cohort_result = build_cohort(standardized, config)
    except CountMismatchError as error:
        write_csv(error.attrition, public_dir / "attrition.csv")
        write_csv(
            error.comparison,
            public_dir / "expected_vs_observed_counts.csv",
        )
        attrition_comparison = error.comparison.loc[
            error.comparison.get("category", pd.Series(dtype="string")).eq(
                "attrition"
            )
        ]
        if not attrition_comparison.empty:
            write_csv(
                attrition_comparison,
                public_dir / "expected_vs_observed_attrition_counts.csv",
            )
        write_json(
            {
                "status": "failed",
                "failure_type": "paper_cohort_count_mismatch",
                "error": str(error),
                "dataset": dataset,
                "artifact_classification": "restricted",
                "config_hash": config["_meta"]["config_hash"],
                "mapping_hash": standardized.mapping_hash,
            },
            public_dir / "failed_run_manifest.json",
        )
        raise
    cohort_result.cohort.to_parquet(
        restricted_dir / "base_acute_care_cohort.parquet", index=False
    )
    cohort_result.baseline.to_parquet(restricted_dir / "baseline_X.parquet", index=False)
    write_csv(cohort_result.attrition, public_dir / "attrition.csv")
    if not cohort_result.count_comparison.empty:
        write_csv(
            cohort_result.count_comparison,
            public_dir / "expected_vs_observed_counts.csv",
        )
        attrition_comparison = cohort_result.count_comparison.loc[
            cohort_result.count_comparison["category"].eq("attrition")
        ]
        if not attrition_comparison.empty:
            write_csv(
                attrition_comparison,
                public_dir / "expected_vs_observed_attrition_counts.csv",
            )
    write_json(
        {
            "cohort_hash": cohort_result.cohort_hash,
            "row_order_hash": cohort_result.row_order_hash,
            "visits": len(cohort_result.cohort),
            "patients": int(cohort_result.cohort["patient_id"].nunique()),
            "outcomes": int(cohort_result.cohort["outcome"].sum()),
            "landmark_hours": config["cohort"]["landmark_hours"],
            "predictor_window_hours": config["cohort"]["predictor_window_hours"],
            "outcome_horizon_days": config["cohort"]["outcome_horizon_days"],
            "death_time_precision_counts": {
                str(key): int(value)
                for key, value in cohort_result.cohort[
                    "death_time_precision"
                ].value_counts(dropna=False).items()
            },
            "death_source_conflict_count": int(
                cohort_result.cohort["death_source_conflict"].eq(True).sum()
            ),
        },
        public_dir / "cohort_manifest.json",
    )
    if stop_after == 2:
        return _partial_result(config, public_dir, restricted_dir, standardized)

    # Stage 3: one patient-level fold assignment reused everywhere.
    fold_result = create_patient_folds(cohort_result.cohort, config)
    write_csv(
        fold_result.assignments,
        restricted_dir / "fold_assignments_restricted.csv",
    )
    write_csv(fold_result.public_summary, public_dir / "fold_summary.csv")
    if stop_after == 3:
        return _partial_result(config, public_dir, restricted_dir, standardized)

    # Stages 4-5: semantics validation and first-24-hour event preparation.
    try:
        prepared = prepare_domain_events(standardized, cohort_result.cohort, config)
    except LinkageError as error:
        if isinstance(error.diagnostics, pd.DataFrame) and not error.diagnostics.empty:
            write_csv(
                error.diagnostics,
                restricted_dir / "event_linkage_failure_diagnostics.csv",
            )
        raise
    write_csv(prepared.audit, public_dir / "event_linkage_audit.csv")
    write_csv(
        prepared.restricted_audit,
        restricted_dir / "event_linkage_restricted.csv",
    )
    _enforce_expected_event_counts(
        config,
        prepared.audit,
        public_dir,
        standardized.mapping_hash,
    )
    for domain, frame in prepared.events.items():
        frame.to_parquet(restricted_dir / f"prepared_{domain}.parquet", index=False)
    write_json(
        {
            "mapping_hash": standardized.mapping_hash,
            "domains": {
                domain: {
                    "qualifying_events": len(frame),
                    "source_tables": sorted(frame["source_table"].astype(str).unique().tolist()),
                    "semantics": sorted(frame["semantics"].dropna().astype(str).unique().tolist()),
                    "linkage_strategies": sorted(
                        frame["linkage_strategy"].dropna().astype(str).unique().tolist()
                    ),
                }
                for domain, frame in prepared.events.items()
            },
            "mimic_native_rules": config.get("source", {}).get("native", {}),
        },
        public_dir / "mapping_validation.json",
    )
    if stop_after in {4, 5}:
        return _partial_result(config, public_dir, restricted_dir, standardized)

    assignments = fold_result.assignments.set_index("cohort_visit_number")
    cohort_indexed = cohort_result.cohort.set_index("cohort_visit_number")
    predictions_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    unit_audits: list[pd.DataFrame] = []
    fit_manifests: list[dict[str, Any]] = []
    importance_parts: list[pd.DataFrame] = []
    feature_manifest_rows: list[dict[str, Any]] = []
    derived_selection_parts: list[pd.DataFrame] = []
    feature_dictionary_rows: list[dict[str, Any]] = []
    matrix_manifest_rows: list[dict[str, Any]] = []
    matrix_names = list(config["matrices"])
    model_names = list(config["models"]["frozen_order"])

    # Stages 6-7: fold selections, feature construction, 160 fits, held-out prediction.
    for fold in range(int(config["folds"]["count"])):
        training_numbers = set(
            assignments.index[assignments["fold"] != fold].astype(int).tolist()
        )
        validation_numbers = assignments.index[assignments["fold"] == fold].astype(int).tolist()
        domains = {}
        for domain in ("measurements", "medications", "procedures"):
            selection = select_concepts(
                prepared.events[domain], training_numbers, domain, fold, config
            )
            feature = build_fold_domain_features(
                cohort_result.cohort,
                selection,
                prepared.events[domain],
                training_numbers,
                config,
            )
            domains[domain] = feature
            fold_feature_dir = restricted_dir / "fold_features" / f"fold_{fold}"
            fold_feature_dir.mkdir(parents=True, exist_ok=True)
            feature.frame.to_parquet(
                fold_feature_dir / f"{domain}.parquet", index=False
            )
            selected = selection.selected.copy()
            selected["selection_hash"] = selection.selection_hash
            selection_parts.append(selected)
            derived_selection_parts.append(feature.selection_audit.copy())
            if not selection.unit_audit.empty:
                audit = selection.unit_audit.copy()
                audit.insert(0, "fold", fold)
                unit_audits.append(audit)
            feature_manifest_rows.append(
                {
                    "fold": fold,
                    "domain": domain,
                    "feature_count": len(feature.feature_names),
                    "constructed_feature_count": feature.full_feature_count,
                    "feature_schema_hash": feature.feature_schema_hash,
                    "feature_value_hash": feature.feature_value_hash,
                    "derived_selection_hash": feature.selection_audit[
                        "derived_selection_hash"
                    ].iloc[0],
                    "selection_hash": selection.selection_hash,
                }
            )
            feature_dictionary_rows.extend(
                {
                    "fold": fold,
                    "domain": domain,
                    "feature_name": feature_name,
                    "selection_hash": selection.selection_hash,
                    "derived_selection_hash": feature.selection_audit[
                        "derived_selection_hash"
                    ].iloc[0],
                }
                for feature_name in feature.feature_names
            )
        if stop_after == 6:
            continue
        for matrix_name in matrix_names:
            matrix = assemble_matrix(
                cohort_result.baseline, domains, matrix_name, config
            )
            train_index = sorted(training_numbers)
            validation_index = validation_numbers
            x_train = matrix.loc[train_index]
            x_validation = matrix.loc[validation_index]
            y_train = cohort_indexed.loc[train_index, "outcome"]
            matrix_manifest_rows.append(
                {
                    "fold": fold,
                    "matrix": matrix_name,
                    "rows": len(matrix),
                    "input_feature_count": int(matrix.shape[1]),
                    "feature_schema_hash": hash_frame_schema(matrix),
                    "feature_matrix_hash": hash_frame_values(
                        matrix.reset_index(),
                        identity_columns=["cohort_visit_number"],
                    ),
                }
            )
            for model_name in model_names:
                fit = fit_predict_fold(
                    x_train, y_train, x_validation, model_name, config
                )
                prediction = pd.DataFrame(
                    {
                        "cohort_visit_number": validation_index,
                        "matrix": matrix_name,
                        "model": model_name,
                        "fold": fold,
                        "probability": fit.probabilities,
                    }
                )
                predictions_parts.append(prediction)
                manifest = {
                    "fold": fold,
                    "matrix": matrix_name,
                    "training_visit_hash": hash_object(train_index),
                    "validation_visit_hash": hash_object(validation_index),
                    "preprocessing_fit_partition_hash": hash_object(train_index),
                    **fit.manifest,
                }
                fit_manifests.append(manifest)
                importance = fit.feature_importance.copy()
                importance.insert(0, "fold", fold)
                importance.insert(1, "matrix", matrix_name)
                importance_parts.append(importance)

    selections = pd.concat(selection_parts, ignore_index=True)
    derived_selections = pd.concat(derived_selection_parts, ignore_index=True)
    selection_target = (
        public_dir / "fold_concept_selections.csv"
        if config.get("synthetic")
        else restricted_dir / "fold_concept_selections.csv"
    )
    write_csv(selections, selection_target)
    derived_selection_target = (
        public_dir / "fold_derived_feature_selections.csv"
        if config.get("synthetic")
        else restricted_dir / "fold_derived_feature_selections.csv"
    )
    write_csv(derived_selections, derived_selection_target)
    selection_frequency = (
        selections.groupby(["domain", "concept_key", "concept_name"], dropna=False)
        .agg(folds_selected=("fold", "nunique"), mean_training_prevalence=("training_visit_prevalence", "mean"))
        .reset_index()
        .sort_values(["domain", "folds_selected", "concept_key"], ascending=[True, False, True])
    )
    frequency_target = (
        public_dir / "concept_selection_frequency.csv"
        if config.get("synthetic")
        else restricted_dir / "concept_selection_frequency.csv"
    )
    write_csv(selection_frequency, frequency_target)
    unit_audit = (
        pd.concat(unit_audits, ignore_index=True)
        if unit_audits
        else pd.DataFrame(
            columns=["fold", "concept_key", "unit", "status", "count"]
        )
    )
    unit_audit_target = (
        public_dir / "measurement_unit_audit.csv"
        if config.get("synthetic")
        else restricted_dir / "measurement_unit_audit.csv"
    )
    write_csv(unit_audit, unit_audit_target)
    write_csv(pd.DataFrame(feature_manifest_rows), public_dir / "feature_manifest.csv")
    feature_dictionary_target = (
        public_dir / "feature_dictionary.csv"
        if config.get("synthetic")
        else restricted_dir / "feature_dictionary.csv"
    )
    write_csv(pd.DataFrame(feature_dictionary_rows), feature_dictionary_target)
    write_json(
        {
            "fold_hash": fold_result.fold_hash,
            "selection_hashes": sorted(selections["selection_hash"].unique().tolist()),
            "folds": int(config["folds"]["count"]),
        },
        public_dir / "manifests" / "fold_manifest.json",
    )
    write_json(
        {
            "domains": feature_manifest_rows,
            "expected_counts": {
                domain: config["features"][domain]["expected_count"]
                for domain in ("measurements", "medications", "procedures")
            },
        },
        public_dir / "manifests" / "domain_manifest.json",
    )
    write_json(
        {
            "selection_count": len(selections),
            "selection_hashes": sorted(selections["selection_hash"].unique().tolist()),
            "ranking": config["features"]["ranking"],
            "tie_break": config["features"]["tie_break"],
            "derived_feature_selection_count": len(derived_selections),
            "derived_selection_hashes": sorted(
                derived_selections["derived_selection_hash"].unique().tolist()
            ),
            "derived_feature_ranking": config["features"][
                "derived_feature_ranking"
            ],
            "derived_feature_tie_break": config["features"][
                "derived_feature_tie_break"
            ],
            "retained_per_domain_per_fold": config["features"][
                "retained_derived_feature_count"
            ],
            "training_folds_only": True,
        },
        public_dir / "manifests" / "selection_manifest.json",
    )
    _enforce_expected_selection_counts(
        config,
        selections,
        derived_selections,
        public_dir,
        standardized.mapping_hash,
    )
    if stop_after == 6:
        return _partial_result(config, public_dir, restricted_dir, standardized)

    predictions = pd.concat(predictions_parts, ignore_index=True)
    validate_oof_predictions(
        predictions,
        cohort_result.cohort,
        fold_result.assignments,
        matrix_names,
        model_names,
        fit_manifests,
    )
    if len(fit_manifests) != 160:
        raise IntegrityError(f"Expected 160 outer-fold fits; observed {len(fit_manifests)}")
    write_csv(predictions, restricted_dir / "oof_predictions_restricted.csv")
    importance = pd.concat(importance_parts, ignore_index=True)
    importance_target = (
        public_dir / "fold_feature_importance.csv"
        if config.get("synthetic")
        else restricted_dir / "fold_feature_importance.csv"
    )
    write_csv(importance, importance_target)
    write_csv(pd.DataFrame(matrix_manifest_rows), public_dir / "matrix_manifest.csv")
    write_json(
        {
            "matrix_count": len(matrix_names),
            "definitions": config["matrices"],
            "fold_matrices": matrix_manifest_rows,
        },
        public_dir / "manifests" / "matrix_manifest.json",
    )
    write_json(
        {
            "fit_count": len(fit_manifests),
            "models": model_names,
            "matrices": matrix_names,
            "fits": fit_manifests,
        },
        public_dir / "manifests" / "model_manifest.json",
    )
    if stop_after == 7:
        return _partial_result(config, public_dir, restricted_dir, standardized)

    # Stage 8: all aggregate OOF analyses and manifests.
    evaluation_result = evaluate_predictions(predictions, cohort_result.cohort, config)
    for filename, table in evaluation_result.tables.items():
        write_csv(table, public_dir / filename)
    if config["models"].get("shap", {}).get("enabled", False):
        shap_folds, shap_summary = _build_shap_outputs(
            config,
            cohort_result,
            fold_result.assignments,
            restricted_dir,
            evaluation_result.tables["best_model_by_matrix.csv"],
            derived_selections,
        )
        shap_fold_target = (
            public_dir / "shap_fold_aggregates.csv"
            if config.get("synthetic")
            else restricted_dir / "shap_fold_aggregates.csv"
        )
        write_csv(shap_folds, shap_fold_target)
        write_csv(shap_summary, public_dir / "shap_summary.csv")
    write_json(
        {
            "dataset": dataset,
            "adapter": config["adapter"],
            "input_hashes": standardized.input_hashes,
            "source_release_or_snapshot": config["source"].get(
                "release_or_snapshot", config["source"].get("expected_version")
            ),
            "config_hash": config["_meta"]["config_hash"],
            "mapping_hash": standardized.mapping_hash,
            "input_collection_hash": hash_object(
                {
                    "analytical_table_hashes": standardized.input_hashes,
                    "source_release_or_snapshot": config["source"].get(
                        "release_or_snapshot",
                        config["source"].get("expected_version"),
                    ),
                    "config_hash": config["_meta"]["config_hash"],
                    "mapping_hash": standardized.mapping_hash,
                }
            ),
        },
        public_dir / "manifests" / "input_manifest.json",
    )
    write_json(
        {
            "config_hash": config["_meta"]["config_hash"],
            "mapping_hash": standardized.mapping_hash,
            "cohort_hash": cohort_result.cohort_hash,
            "row_order_hash": cohort_result.row_order_hash,
            "fold_hash": fold_result.fold_hash,
        },
        public_dir / "manifests" / "configuration_manifest.json",
    )
    write_json(
        {
            "dataset": dataset,
            "visits": len(cohort_result.cohort),
            "patients": int(cohort_result.cohort["patient_id"].nunique()),
            "events": int(cohort_result.cohort["outcome"].sum()),
            "folds": 5,
            "matrices": 8,
            "models": 4,
            "model_fits": 160,
            "oof_probabilities_per_visit": 32,
            "retained_features_per_domain": 21,
            "bootstrap_repetitions": evaluation_result.bootstrap_repetitions,
        },
        public_dir / "manifests" / "dataset_manifest.json",
    )
    software = _software_versions()
    code_hash = _code_hash()
    output_hashes = _public_output_hashes(public_dir)
    write_json(output_hashes, public_dir / "manifests" / "output_manifest.json")
    release = config.get("paper", {}).get("release_clearance", {})
    classification = (
        "public_synthetic"
        if config.get("synthetic")
        else str(release.get("classification", "restricted"))
    )
    restricted_artifacts = [
        "base_acute_care_cohort.parquet",
        "baseline_X.parquet",
        "fold_assignments_restricted.csv",
        "event_linkage_restricted.csv",
        "prepared_measurements.parquet",
        "prepared_medications.parquet",
        "prepared_procedures.parquet",
        "fold_features/fold_<k>/<domain>.parquet",
        "oof_predictions_restricted.csv",
    ]
    if not config.get("synthetic"):
        restricted_artifacts.extend(
            [
                "fold_concept_selections.csv",
                "fold_derived_feature_selections.csv",
                "concept_selection_frequency.csv",
                "measurement_unit_audit.csv",
                "feature_dictionary.csv",
                "fold_feature_importance.csv",
                "shap_fold_aggregates.csv",
            ]
        )
    run_manifest = {
        "run_id": hash_object(
            {
                "dataset": dataset,
                "config": config["_meta"]["config_hash"],
                "code": code_hash,
                "inputs": standardized.input_hashes,
                "cohort": cohort_result.cohort_hash,
                "folds": fold_result.fold_hash,
                "selections": sorted(selections["selection_hash"].unique()),
            }
        ),
        "dataset": dataset,
        "adapter": config["adapter"],
        "created_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_is_dirty(),
        "code_hash": code_hash,
        "config_hash": config["_meta"]["config_hash"],
        "mapping_hash": standardized.mapping_hash,
        "cohort_hash": cohort_result.cohort_hash,
        "row_order_hash": cohort_result.row_order_hash,
        "fold_hash": fold_result.fold_hash,
        "input_hashes": standardized.input_hashes,
        "output_hashes": output_hashes,
        "software_versions": software,
        "warnings": [],
        "failures": [],
        "artifact_classification": classification,
        "privacy_gate": {
            "classification": classification,
            "ran": False,
            "passed": False,
            "release_approved": bool(release.get("approved")),
            "small_cell_threshold": release.get(
                "small_cell_threshold",
                config.get("privacy", {}).get("small_cell_threshold"),
            ),
        },
        "paper_reproduction_status": (
            "not_applicable_synthetic"
            if config.get("synthetic")
            else "not_reconciled_or_release_cleared"
        ),
        "restricted_artifacts": restricted_artifacts,
    }
    write_json(run_manifest, public_dir / "run_manifest.json")
    scan_public_tree(
        public_dir,
        classification=classification,
        small_cell_threshold=release.get(
            "small_cell_threshold",
            config.get("privacy", {}).get("small_cell_threshold"),
        ),
        release_approved=bool(release.get("approved")),
    )
    run_manifest["privacy_gate"]["ran"] = True
    run_manifest["privacy_gate"]["passed"] = True
    write_json(run_manifest, public_dir / "run_manifest.json")
    return RunResult(dataset, public_dir, restricted_dir, run_manifest)


def synthetic_run() -> dict[str, Any]:
    """Run both adapters independently against equivalent public synthetic content."""
    output = PROJECT_ROOT / "outputs" / "synthetic"
    restricted = PROJECT_ROOT / "restricted_outputs" / "synthetic"
    results = [
        run_pipeline(
            PROJECT_ROOT / "configs" / "chorus.example.yaml",
            output,
            restricted,
        ),
        run_pipeline(
            PROJECT_ROOT / "configs" / "mimic.example.yaml",
            output,
            restricted,
        ),
    ]
    summary_rows = []
    for result in results:
        manifest = read_json(result.public_dir / "manifests" / "dataset_manifest.json")
        summary_rows.append(manifest)
    summary = pd.DataFrame(summary_rows).sort_values("dataset", kind="stable")
    write_csv(summary, output / "synthetic_run_summary.csv")
    parent = {
        "created_utc": utc_timestamp(),
        "datasets": [result.dataset for result in results],
        "child_run_ids": {result.dataset: result.run_manifest["run_id"] for result in results},
        "summary_hash": hash_file(output / "synthetic_run_summary.csv"),
    }
    write_json(parent, output / "run_manifest.json")
    return parent


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    """Recompute output hashes and enforce the committed synthetic invariants."""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise IntegrityError(f"Run directory does not exist: {root}")
    datasets = ["chorus", "mimiciv"] if (root / "chorus").is_dir() else [root.name]
    verified = {}
    for dataset in datasets:
        dataset_dir = root / dataset if (root / dataset).is_dir() else root
        manifest = read_json(dataset_dir / "run_manifest.json")
        verify_hashes(dataset_dir, manifest["output_hashes"])
        expected_files = set(manifest["output_hashes"]) | {
            "run_manifest.json",
            "manifests/output_manifest.json",
        }
        actual_files = {
            path.relative_to(dataset_dir).as_posix()
            for path in dataset_dir.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise IntegrityError(
                "Run artifact set differs from its manifest: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"unexpected={sorted(actual_files - expected_files)}"
            )
        privacy_gate = manifest.get("privacy_gate", {})
        scan_public_tree(
            dataset_dir,
            classification=str(
                manifest.get("artifact_classification", "restricted")
            ),
            small_cell_threshold=privacy_gate.get("small_cell_threshold"),
            release_approved=bool(privacy_gate.get("release_approved")),
        )
        dataset_manifest = read_json(dataset_dir / "manifests" / "dataset_manifest.json")
        _verify_dataset_invariants(dataset_dir, dataset_manifest)
        _verify_output_schemas(dataset_dir)
        verified[dataset] = {
            "run_id": manifest["run_id"],
            "files_verified": len(manifest["output_hashes"]),
        }
    if set(datasets) == {"chorus", "mimiciv"}:
        expected = read_json(
            PROJECT_ROOT / "synthetic_data" / "expected_outputs" / "expected_summary.json"
        )
        expected_path = (
            PROJECT_ROOT
            / "synthetic_data"
            / "expected_outputs"
            / "expected_aggregate_hashes.json"
        )
        checksum_path = expected_path.with_suffix(".sha256")
        if not checksum_path.is_file():
            raise IntegrityError("Synthetic expected-hash checksum sidecar is missing")
        expected_checksum = checksum_path.read_text(encoding="utf-8").strip()
        if hash_file(expected_path) != expected_checksum:
            raise IntegrityError(
                "Synthetic expected hashes changed outside the intentional freeze procedure"
            )
        expected_hashes = read_json(expected_path)
        summary = pd.read_csv(root / "synthetic_run_summary.csv").sort_values(
            "dataset", kind="stable"
        )
        observed = {
            row.dataset: {
                "visits": int(row.visits),
                "patients": int(row.patients),
                "events": int(row.events),
                "folds": int(row.folds),
                "matrices": int(row.matrices),
                "models": int(row.models),
                "model_fits": int(row.model_fits),
                "oof_probabilities_per_visit": int(row.oof_probabilities_per_visit),
            }
            for row in summary.itertuples(index=False)
        }
        if observed != expected["datasets"]:
            raise IntegrityError(
                f"Synthetic expected summary mismatch: expected={expected['datasets']}, observed={observed}"
            )
        if observed["chorus"] != observed["mimiciv"]:
            raise IntegrityError("Equivalent synthetic adapters did not converge to the same cohort")
        if expected_hashes.get("format_version") != 3:
            raise IntegrityError("Unsupported synthetic freeze manifest format")
        for dataset in ("chorus", "mimiciv"):
            frozen = expected_hashes["datasets"][dataset]
            actual_names = sorted(
                path.relative_to(root / dataset).as_posix()
                for path in (root / dataset).rglob("*")
                if path.is_file() and path.name != "run_manifest.json"
            )
            if actual_names != frozen["artifact_names"]:
                raise IntegrityError(
                    f"Synthetic artifact set changed for {dataset}: "
                    f"expected={frozen['artifact_names']}, observed={actual_names}"
                )
            observed_hashes = {
                relative: _canonical_artifact_hash(root / dataset / relative)
                for relative in frozen["artifact_names"]
            }
            if observed_hashes != frozen["artifact_hashes"]:
                changed = sorted(
                    name
                    for name in set(observed_hashes) | set(frozen["artifact_hashes"])
                    if observed_hashes.get(name) != frozen["artifact_hashes"].get(name)
                )
                raise IntegrityError(
                    f"Frozen synthetic analytical artifacts changed for "
                    f"{dataset}: {changed}"
                )
            safe_manifest = _safe_run_manifest(
                read_json(root / dataset / "run_manifest.json")
            )
            if hash_object(safe_manifest) != frozen["safe_run_manifest_hash"]:
                raise IntegrityError(
                    f"Safe run-manifest fields changed for synthetic {dataset}"
                )
        actual_summary_hash = _canonical_artifact_hash(
            root / "synthetic_run_summary.csv"
        )
        if actual_summary_hash != expected_hashes["synthetic_run_summary_hash"]:
            raise IntegrityError("Synthetic aggregate summary hash does not match the release")
        parent_safe = _safe_parent_manifest(read_json(root / "run_manifest.json"))
        if hash_object(parent_safe) != expected_hashes["safe_parent_manifest_hash"]:
            raise IntegrityError("Safe synthetic overall manifest fields changed")
    return {"verified": verified, "status": "ok"}


def freeze_synthetic_expected(
    run_dir: str | Path,
    *,
    approve_update: bool,
) -> dict[str, Any]:
    """Intentional procedure for updating all deterministic synthetic pins."""
    if not approve_update:
        raise ConfigurationError(
            "Synthetic freeze requires --approve-update after reviewing all aggregate changes"
        )
    root = Path(run_dir).resolve()
    if not all((root / dataset).is_dir() for dataset in ("chorus", "mimiciv")):
        raise IntegrityError("Freeze requires completed CHoRUS and MIMIC synthetic runs")
    frozen: dict[str, Any] = {
        "format_version": 3,
        "canonical_float_decimal_places": 10,
        "datasets": {},
    }
    for dataset in ("chorus", "mimiciv"):
        dataset_dir = root / dataset
        manifest = read_json(dataset_dir / "run_manifest.json")
        verify_hashes(dataset_dir, manifest["output_hashes"])
        artifact_names = sorted(
            path.relative_to(dataset_dir).as_posix()
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.name != "run_manifest.json"
        )
        frozen["datasets"][dataset] = {
            "artifact_names": artifact_names,
            "artifact_hashes": {
                relative: _canonical_artifact_hash(dataset_dir / relative)
                for relative in artifact_names
            },
            "safe_run_manifest_hash": hash_object(_safe_run_manifest(manifest)),
        }
    frozen["synthetic_run_summary_hash"] = _canonical_artifact_hash(
        root / "synthetic_run_summary.csv"
    )
    frozen["safe_parent_manifest_hash"] = hash_object(
        _safe_parent_manifest(read_json(root / "run_manifest.json"))
    )
    expected_path = (
        PROJECT_ROOT
        / "synthetic_data"
        / "expected_outputs"
        / "expected_aggregate_hashes.json"
    )
    write_json(frozen, expected_path)
    expected_path.with_suffix(".sha256").write_text(
        hash_file(expected_path) + "\n", encoding="utf-8"
    )
    return {
        "status": "ok",
        "datasets": ["chorus", "mimiciv"],
        "artifacts_frozen": {
            dataset: len(frozen["datasets"][dataset]["artifact_names"])
            for dataset in ("chorus", "mimiciv")
        },
        "expected_manifest_sha256": hash_file(expected_path),
    }


def _canonical_artifact_hash(path: Path) -> str:
    """Hash a public artifact using the frozen cross-platform contract."""
    relative_name = path.as_posix()
    if path.name == "output_manifest.json":
        raw_manifest = read_json(path)
        return hash_object(
            {
                relative: _canonical_artifact_hash(path.parent.parent / relative)
                for relative in sorted(raw_manifest)
                if (path.parent.parent / relative).is_file()
            }
        )
    if path.suffix.lower() == ".json":
        return hash_object(read_json(path))
    if path.suffix.lower() == ".csv" or relative_name.endswith(".csv.gz"):
        frame = pd.read_csv(path)
        rows: list[list[Any]] = []
        for values in frame.itertuples(index=False, name=None):
            row: list[Any] = []
            for value in values:
                if pd.isna(value):
                    row.append(None)
                elif isinstance(value, bool | np.bool_):
                    row.append(bool(value))
                elif isinstance(value, float | np.floating):
                    numeric = float(value)
                    if np.isposinf(numeric):
                        row.append("+Infinity")
                    elif np.isneginf(numeric):
                        row.append("-Infinity")
                    else:
                        rounded = round(numeric, 10)
                        row.append(0.0 if rounded == 0 else rounded)
                elif isinstance(value, int | np.integer):
                    row.append(int(value))
                else:
                    row.append(str(value))
            rows.append(row)
        return hash_object(
            {
                "columns": [str(column) for column in frame.columns],
                "rows": rows,
            }
        )
    return hash_file(path)


def _safe_run_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "created_utc",
        "git_commit",
        "git_worktree_dirty",
        "output_hashes",
    }
    return {
        key: value
        for key, value in manifest.items()
        if key not in excluded
    }


def _safe_parent_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"created_utc", "summary_hash"}
    }


def verify_paper_run(
    config_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Fail closed unless an actual paper run and governance status reconcile."""
    config = load_config(config_path)
    if not config.get("paper_run"):
        raise ConfigurationError("Paper verification requires paper_run: true")
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise IntegrityError(f"Paper run directory does not exist: {root}")
    manifest = read_json(root / "run_manifest.json")
    dataset_manifest = read_json(root / "manifests" / "dataset_manifest.json")
    cohort_manifest = read_json(root / "cohort_manifest.json")
    mapping_manifest = read_json(root / "manifests" / "mapping_manifest.json")
    input_manifest = read_json(root / "manifests" / "input_manifest.json")
    model_manifest = read_json(root / "manifests" / "model_manifest.json")
    failures: list[str] = []
    restricted_root = root.parent
    if manifest.get("config_hash") != config["_meta"]["config_hash"]:
        failures.append("configuration hash differs from the executed run")
    if manifest.get("mapping_hash") != mapping_manifest.get("mapping_hash"):
        failures.append("mapping hash is internally inconsistent")
    if input_manifest.get("source_release_or_snapshot") != config["source"].get(
        "release_or_snapshot"
    ):
        failures.append("source release/snapshot differs from the paper configuration")
    expected = config["cohort"]["expected_counts"]
    observed = {
        "visits": cohort_manifest.get("visits"),
        "patients": cohort_manifest.get("patients"),
        "deaths": cohort_manifest.get("outcomes"),
    }
    if observed != expected:
        failures.append(f"cohort counts differ: expected={expected}, observed={observed}")
    if dataset_manifest.get("folds") != 5:
        failures.append("five frozen folds were not recorded")
    if dataset_manifest.get("matrices") != 8 or dataset_manifest.get("models") != 4:
        failures.append("eight matrices and four models were not recorded")
    if model_manifest.get("matrices") != list(config["matrices"]):
        failures.append("executed feature matrices differ from the frozen configuration")
    if model_manifest.get("models") != list(config["models"]["frozen_order"]):
        failures.append("executed model order differs from the frozen configuration")
    try:
        cohort = pd.read_parquet(
            restricted_root / "base_acute_care_cohort.parquet"
        )
        assignments = pd.read_csv(
            restricted_root / "fold_assignments_restricted.csv"
        )
        predictions = pd.read_csv(
            restricted_root / "oof_predictions_restricted.csv"
        )
        validate_oof_predictions(
            predictions,
            cohort,
            assignments,
            list(config["matrices"]),
            list(config["models"]["frozen_order"]),
            model_manifest.get("fits"),
        )
    except (FileNotFoundError, IntegrityError, KeyError, ValueError) as error:
        failures.append(f"restricted OOF/fold evidence failed verification: {error}")
    try:
        _verify_selection_artifacts(config, root)
    except IntegrityError as error:
        failures.append(str(error))
    required = {
        "fold_metrics.csv",
        "pooled_oof_metrics.csv",
        "best_model_by_matrix.csv",
        "selected_models_calibration_coordinates.csv",
        "selected_models_sensitivity_at_90_specificity.csv",
        "selected_models_top_10_percent_risk_analysis.csv",
        "selected_models_decision_curve_coordinates.csv",
        "prespecified_paired_matrix_comparisons.csv",
        "shap_summary.csv",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        failures.append(f"required aggregate outputs are missing: {missing}")
    event_comparison_path = root / "expected_vs_observed_event_counts.csv"
    if not event_comparison_path.is_file():
        failures.append("expected-versus-observed event-count diagnostics are missing")
    else:
        event_comparison = pd.read_csv(event_comparison_path)
        if "matches" not in event_comparison or not _strict_boolean(
            event_comparison["matches"], "event-count matches"
        ).all():
            failures.append("configured paper event counts do not match observed counts")
        else:
            try:
                event_audit = pd.read_csv(root / "event_linkage_audit.csv")
                recomputed, missing_targets, unexpected_targets = (
                    _expected_event_count_comparison(config, event_audit)
                )
                if missing_targets or unexpected_targets:
                    raise IntegrityError(
                        "configured paper event-count targets are incomplete"
                    )
                _assert_count_comparison_equal(
                    event_comparison,
                    recomputed,
                    ["domain", "attrition_stage"],
                    "event-count comparison",
                )
            except (FileNotFoundError, IntegrityError) as error:
                failures.append(str(error))
    count_comparison_path = root / "expected_vs_observed_counts.csv"
    if not count_comparison_path.is_file():
        failures.append("expected-versus-observed cohort-count diagnostics are missing")
    else:
        try:
            _verify_cohort_count_comparison(
                config,
                pd.read_csv(count_comparison_path),
                pd.read_csv(root / "attrition.csv"),
                cohort_manifest,
            )
        except (FileNotFoundError, IntegrityError) as error:
            failures.append(str(error))
    attrition_comparison_path = (
        root / "expected_vs_observed_attrition_counts.csv"
    )
    if not attrition_comparison_path.is_file():
        failures.append(
            "expected-versus-observed attrition-count diagnostics are missing"
        )
    else:
        attrition_comparison = pd.read_csv(attrition_comparison_path)
        if "matches" not in attrition_comparison or not _strict_boolean(
            attrition_comparison["matches"], "attrition-count matches"
        ).all():
            failures.append(
                "configured paper attrition counts do not match observed counts"
            )
    selection_comparison_path = (
        root / "expected_vs_observed_selection_counts.csv"
    )
    if not selection_comparison_path.is_file():
        failures.append("expected-versus-observed selection-count diagnostics are missing")
    else:
        selection_comparison = pd.read_csv(selection_comparison_path)
        if "matches" not in selection_comparison or not _strict_boolean(
            selection_comparison["matches"], "selection-count matches"
        ).all():
            failures.append(
                "configured paper selection counts do not match observed counts"
            )
        else:
            try:
                _verify_selection_count_comparison(
                    config, selection_comparison
                )
            except IntegrityError as error:
                failures.append(str(error))
    if not (restricted_root / "measurement_unit_audit.csv").is_file():
        failures.append("restricted measurement-unit audit is missing")
    paper = config["paper"]
    if not paper.get("manuscript_reconciled"):
        failures.append("manuscript reconciliation has not been approved")
    if not paper.get("release_clearance", {}).get("approved"):
        failures.append("aggregate release clearance has not been approved")
    privacy_gate = manifest.get("privacy_gate", {})
    if (
        privacy_gate.get("classification") != "public_clinical"
        or not privacy_gate.get("ran")
        or not privacy_gate.get("passed")
    ):
        failures.append("the public_clinical privacy gate did not run and pass")
    else:
        try:
            scan_public_tree(
                root,
                classification="public_clinical",
                small_cell_threshold=privacy_gate.get("small_cell_threshold"),
                release_approved=bool(privacy_gate.get("release_approved")),
            )
        except IntegrityError as error:
            failures.append(str(error))
    if failures:
        raise IntegrityError("Paper verification failed closed: " + "; ".join(failures))
    verify_run(root)
    return {
        "status": "ok",
        "dataset": config["dataset"],
        "run_id": manifest["run_id"],
        "source_release_or_snapshot": config["source"]["release_or_snapshot"],
        "manuscript_reconciled": True,
        "release_cleared": True,
    }


def _enforce_expected_event_counts(
    config: dict[str, Any],
    audit: pd.DataFrame,
    public_dir: Path,
    mapping_hash: str,
) -> None:
    """Persist observed event diagnostics before a configured mismatch fails."""
    expected = config.get("paper", {}).get("expected_event_counts")
    if not config.get("paper_run"):
        return
    if not isinstance(expected, dict):
        raise ConfigurationError(
            "Paper event counts must be confirmed before source execution"
        )
    comparison, missing, unexpected = _expected_event_count_comparison(
        config, audit
    )
    write_csv(comparison, public_dir / "expected_vs_observed_event_counts.csv")
    if missing or unexpected or not comparison["matches"].all():
        write_json(
            {
                "status": "failed",
                "failure_type": "paper_event_count_mismatch",
                "dataset": config["dataset"],
                "artifact_classification": "restricted",
                "config_hash": config["_meta"]["config_hash"],
                "mapping_hash": mapping_hash,
                "missing_expected_counts": missing,
                "unexpected_expected_counts": unexpected,
            },
            public_dir / "failed_run_manifest.json",
        )
        raise IntegrityError(
            "Paper event-count mismatch; persisted expected-versus-observed diagnostics"
        )


def _expected_event_count_comparison(
    config: dict[str, Any],
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    paper = config.get("paper", {})
    expected = paper.get("expected_event_counts", {})
    default_tolerance = int(
        paper.get("expected_count_tolerances", {}).get("default", 0)
    )
    required_stages = list(
        paper.get("expected_event_count_stages", ["qualifying"])
    )
    domains = ("measurements", "medications", "procedures")
    targets: dict[tuple[str, str], Any] = {}
    for key, value in expected.items():
        domain, separator, status = str(key).partition(".")
        targets[(domain, status if separator else "qualifying")] = value
    required = {
        (domain, status) for domain in domains for status in required_stages
    }
    missing = sorted(
        f"{domain}.{status}" for domain, status in required - set(targets)
    )
    unexpected = sorted(
        f"{domain}.{status}"
        for domain, status in set(targets) - required
    )
    rows: list[dict[str, Any]] = []
    for domain_name, status in sorted(required):
        observed_rows = audit.loc[
            audit["domain"].eq(domain_name) & audit["status"].eq(status),
            "count",
        ]
        observed = int(observed_rows.sum()) if not observed_rows.empty else 0
        target = targets.get((domain_name, status))
        if isinstance(target, dict):
            expected_value = target.get("expected")
            tolerance = int(target.get("tolerance", default_tolerance))
        else:
            expected_value = target
            tolerance = default_tolerance
        difference = (
            abs(observed - int(expected_value))
            if expected_value is not None
            else pd.NA
        )
        rows.append(
            {
                "domain": domain_name,
                "attrition_stage": status,
                "expected": (
                    int(expected_value) if expected_value is not None else pd.NA
                ),
                "observed": observed,
                "absolute_difference": difference,
                "tolerance": tolerance,
                "matches": bool(
                    expected_value is not None and difference <= tolerance
                ),
            }
        )
    return pd.DataFrame(rows), missing, unexpected


def _enforce_expected_selection_counts(
    config: dict[str, Any],
    concepts: pd.DataFrame,
    derived: pd.DataFrame,
    public_dir: Path,
    mapping_hash: str,
) -> None:
    """Persist and enforce actual fold/domain concept and feature counts."""
    if not config.get("paper_run"):
        return
    expected = config.get("paper", {}).get("expected_selection_counts")
    if not isinstance(expected, dict):
        raise ConfigurationError(
            "Paper selection-count expectations must be confirmed before execution"
        )
    default_tolerance = int(
        config.get("paper", {})
        .get("expected_count_tolerances", {})
        .get("default", 0)
    )
    required = {
        "selected_concepts_per_fold_domain",
        "selected_features_per_fold_domain",
        "candidate_measurements",
        "candidate_medications",
        "candidate_procedures",
    }
    if set(expected) != required:
        raise ConfigurationError(
            "Paper selection-count expectations have missing or unexpected keys"
        )
    rows: list[dict[str, Any]] = []
    for fold in range(int(config["folds"]["count"])):
        for domain in ("measurements", "medications", "procedures"):
            domain_concepts = concepts.loc[
                concepts["fold"].eq(fold) & concepts["domain"].eq(domain)
            ]
            domain_derived = derived.loc[
                derived["fold"].eq(fold) & derived["domain"].eq(domain)
            ].copy()
            domain_derived["selected"] = _strict_boolean(
                domain_derived["selected"], "derived-feature selected"
            )
            measures = {
                "selected_concepts": (
                    "selected_concepts_per_fold_domain",
                    len(domain_concepts),
                ),
                "candidate_features": (
                    f"candidate_{domain}",
                    len(domain_derived),
                ),
                "selected_features": (
                    "selected_features_per_fold_domain",
                    int(domain_derived["selected"].sum()),
                ),
            }
            for measure, (target_key, observed) in measures.items():
                target = expected[target_key]
                if isinstance(target, dict):
                    expected_value = int(target["expected"])
                    tolerance = int(
                        target.get("tolerance", default_tolerance)
                    )
                else:
                    expected_value = int(target)
                    tolerance = default_tolerance
                difference = abs(int(observed) - expected_value)
                rows.append(
                    {
                        "fold": fold,
                        "domain": domain,
                        "measure": measure,
                        "expected": expected_value,
                        "observed": int(observed),
                        "absolute_difference": difference,
                        "tolerance": tolerance,
                        "matches": difference <= tolerance,
                    }
                )
    comparison = pd.DataFrame(rows)
    write_csv(
        comparison,
        public_dir / "expected_vs_observed_selection_counts.csv",
    )
    if not comparison["matches"].all():
        write_json(
            {
                "status": "failed",
                "failure_type": "paper_selection_count_mismatch",
                "dataset": config["dataset"],
                "artifact_classification": "restricted",
                "config_hash": config["_meta"]["config_hash"],
                "mapping_hash": mapping_hash,
            },
            public_dir / "failed_run_manifest.json",
        )
        raise IntegrityError(
            "Paper selection-count mismatch; persisted expected-versus-observed "
            "diagnostics"
        )


def _verify_selection_count_comparison(
    config: dict[str, Any],
    comparison: pd.DataFrame,
) -> None:
    expected_config = config["paper"]["expected_selection_counts"]
    target_keys = {
        "selected_concepts": "selected_concepts_per_fold_domain",
        "selected_features": "selected_features_per_fold_domain",
        "candidate_features": None,
    }
    expected_rows = int(config["folds"]["count"]) * 3 * len(target_keys)
    keys = ["fold", "domain", "measure"]
    if (
        len(comparison) != expected_rows
        or comparison.duplicated(keys, keep=False).any()
    ):
        raise IntegrityError(
            "selection-count comparison lacks complete unique fold/domain evidence"
        )
    for row in comparison.itertuples(index=False):
        if row.measure == "candidate_features":
            target = int(expected_config[f"candidate_{row.domain}"])
        elif row.measure in target_keys:
            target = int(expected_config[target_keys[row.measure]])
        else:
            raise IntegrityError(
                f"selection-count comparison has unknown measure {row.measure!r}"
            )
        if (
            int(row.expected) != target
            or int(row.observed) != target
            or int(row.absolute_difference) != 0
        ):
            raise IntegrityError(
                "selection-count comparison differs from verified selection evidence"
            )


def _verify_cohort_count_comparison(
    config: dict[str, Any],
    comparison: pd.DataFrame,
    attrition: pd.DataFrame,
    cohort_manifest: dict[str, Any],
) -> None:
    required_columns = {
        "category",
        "stage",
        "measure",
        "expected",
        "observed",
        "absolute_difference",
        "tolerance",
        "matches",
    }
    if required_columns - set(comparison):
        raise IntegrityError("cohort-count comparison is missing evidence columns")
    targets: dict[tuple[str, str, str], tuple[int, int]] = {}
    default_tolerance = int(
        config["paper"]["expected_count_tolerances"]["default"]
    )
    expected_final = config["cohort"]["expected_counts"]
    for measure in ("visits", "patients", "deaths"):
        targets[("final_cohort", "final eligible cohort", measure)] = (
            int(expected_final[measure]),
            default_tolerance,
        )
    for step, measures in config["paper"]["expected_attrition_counts"].items():
        for measure in ("visits", "patients"):
            value = measures[measure]
            if isinstance(value, dict):
                target = int(value["expected"])
                tolerance = int(value.get("tolerance", default_tolerance))
            else:
                target = int(value)
                tolerance = default_tolerance
            targets[("attrition", step, measure)] = (target, tolerance)
    observed_attrition = attrition.set_index("step")
    observed_final = {
        "visits": int(cohort_manifest["visits"]),
        "patients": int(cohort_manifest["patients"]),
        "deaths": int(cohort_manifest["outcomes"]),
    }
    if len(comparison) != len(targets):
        raise IntegrityError("cohort-count comparison has incomplete stage evidence")
    observed_keys = set()
    for row in comparison.itertuples(index=False):
        key = (str(row.category), str(row.stage), str(row.measure))
        if key in observed_keys or key not in targets:
            raise IntegrityError(
                "cohort-count comparison has duplicate or unknown evidence"
            )
        observed_keys.add(key)
        target, tolerance = targets[key]
        if key[0] == "final_cohort":
            observed = observed_final[key[2]]
        else:
            if key[1] not in observed_attrition.index:
                raise IntegrityError(
                    "cohort-count comparison references a missing attrition stage"
                )
            observed = int(observed_attrition.loc[key[1], key[2]])
        difference = abs(observed - target)
        if (
            int(row.expected) != target
            or int(row.observed) != observed
            or int(row.absolute_difference) != difference
            or int(row.tolerance) != tolerance
            or bool(
                _strict_boolean(
                    pd.Series([row.matches]), "cohort-count matches"
                ).iloc[0]
            )
            != (difference <= tolerance)
        ):
            raise IntegrityError(
                "cohort-count comparison differs from configured and observed evidence"
            )


def _build_shap_outputs(
    config: dict[str, Any],
    cohort_result: Any,
    assignments: pd.DataFrame,
    restricted_dir: Path,
    selected_models: pd.DataFrame,
    derived_selections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit selected models and aggregate held-out permutation SHAP by fold."""
    assignment_index = assignments.set_index("cohort_visit_number")
    outcome = cohort_result.cohort.set_index("cohort_visit_number")["outcome"]
    fold_parts: list[pd.DataFrame] = []
    selected = selected_models.set_index("matrix")["model"].to_dict()
    for fold in range(int(config["folds"]["count"])):
        fold_provenance = (
            derived_selections.loc[
                derived_selections["fold"].eq(fold)
                & derived_selections["selected"],
                [
                    "candidate_feature_name",
                    "domain",
                    "source_concept",
                    "summary_type",
                ],
            ]
            .drop_duplicates("candidate_feature_name")
            .set_index("candidate_feature_name")
        )
        domains = {
            domain: pd.read_parquet(
                restricted_dir
                / "fold_features"
                / f"fold_{fold}"
                / f"{domain}.parquet"
            )
            for domain in ("measurements", "medications", "procedures")
        }
        for matrix_name, components in config["matrices"].items():
            matrix = cohort_result.baseline.copy()
            for component in components:
                matrix = matrix.merge(
                    domains[component],
                    on="cohort_visit_number",
                    how="left",
                    validate="one_to_one",
                    sort=False,
                )
            matrix = matrix.set_index("cohort_visit_number", verify_integrity=True)
            training = sorted(
                assignment_index.index[assignment_index["fold"].ne(fold)].astype(int)
            )
            validation = assignment_index.index[
                assignment_index["fold"].eq(fold)
            ].astype(int).tolist()
            shap_part = fold_shap_aggregate(
                matrix.loc[training],
                outcome.loc[training],
                matrix.loc[validation],
                dataset=str(config["dataset"]),
                fold=fold,
                matrix=matrix_name,
                model=str(selected[matrix_name]),
                config=config,
            )
            mapped = shap_part["feature"].map(fold_provenance["domain"])
            clinical = mapped.notna()
            shap_part.loc[clinical, "clinical_domain"] = mapped.loc[clinical]
            shap_part.loc[clinical, "source_concept"] = shap_part.loc[
                clinical, "feature"
            ].map(fold_provenance["source_concept"])
            shap_part.loc[clinical, "summary_type"] = shap_part.loc[
                clinical, "feature"
            ].map(fold_provenance["summary_type"])
            fold_parts.append(shap_part)
    fold_table = pd.concat(fold_parts, ignore_index=True)
    summary = (
        fold_table.groupby(
            [
                "dataset",
                "feature_matrix",
                "model",
                "feature",
                "clinical_domain",
                "source_concept",
                "summary_type",
                "explainer",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            mean_absolute_shap=("mean_absolute_shap", "mean"),
            folds=("outer_fold", "nunique"),
        )
        .sort_values(
            ["feature_matrix", "mean_absolute_shap", "feature"],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    summary["rank"] = (
        summary.groupby("feature_matrix", sort=False).cumcount() + 1
    )
    return fold_table, summary


def _verify_selection_artifacts(config: dict[str, Any], root: Path) -> None:
    """Recompute fold selections, feature tables, and matrix hashes from evidence."""
    restricted_root = root.parent if root.name == "release_candidate_aggregate" else root
    concept_path = restricted_root / "fold_concept_selections.csv"
    derived_path = restricted_root / "fold_derived_feature_selections.csv"
    evidence_paths = {
        "concept selections": concept_path,
        "derived-feature selections": derived_path,
        "cohort": restricted_root / "base_acute_care_cohort.parquet",
        "baseline": restricted_root / "baseline_X.parquet",
        "fold assignments": restricted_root / "fold_assignments_restricted.csv",
        "measurements": restricted_root / "prepared_measurements.parquet",
        "medications": restricted_root / "prepared_medications.parquet",
        "procedures": restricted_root / "prepared_procedures.parquet",
    }
    missing_evidence = [
        label for label, path in evidence_paths.items() if not path.is_file()
    ]
    if missing_evidence:
        raise IntegrityError(
            "restricted selection verification evidence is missing: "
            f"{sorted(missing_evidence)}"
        )
    concepts = pd.read_csv(concept_path)
    derived = pd.read_csv(derived_path)
    cohort = pd.read_parquet(evidence_paths["cohort"])
    baseline = pd.read_parquet(evidence_paths["baseline"])
    assignments = pd.read_csv(evidence_paths["fold assignments"])
    events = {
        domain: pd.read_parquet(evidence_paths[domain])
        for domain in ("measurements", "medications", "procedures")
    }
    required_derived = {
        "fold",
        "domain",
        "rank",
        "candidate_feature_name",
        "source_concept",
        "summary_type",
        "training_support_count",
        "training_support_proportion",
        "selection_score",
        "tie_break_value",
        "training_visit_count",
        "support_definition",
        "selected",
        "selection_rule_identifier",
        "selection_rule_version",
        "derived_selection_hash",
    }
    missing = required_derived - set(derived)
    if missing:
        raise IntegrityError(
            f"derived-feature selection table is missing columns: {sorted(missing)}"
        )
    required_concept = {
        "fold",
        "domain",
        "rank",
        "concept_key",
        "concept_name",
        "training_visit_prevalence",
        "source_table",
        "semantics",
        "units",
        "selected",
        "selection_hash",
    }
    missing_concept = required_concept - set(concepts)
    if missing_concept:
        raise IntegrityError(
            f"concept selection table is missing columns: {sorted(missing_concept)}"
        )
    if assignments["cohort_visit_number"].duplicated().any():
        raise IntegrityError("selection verification found duplicate fold assignments")
    if set(assignments["cohort_visit_number"]) != set(cohort["cohort_visit_number"]):
        raise IntegrityError("selection verification fold assignments do not cover the cohort")

    matrix_manifest_path = root / "matrix_manifest.csv"
    if not matrix_manifest_path.is_file():
        matrix_manifest_path = restricted_root / "release_candidate_aggregate" / "matrix_manifest.csv"
    matrix_manifest = (
        pd.read_csv(matrix_manifest_path)
        if matrix_manifest_path.is_file()
        else pd.DataFrame()
    )
    domains = ("measurements", "medications", "procedures")
    for fold in range(int(config["folds"]["count"])):
        training_visits = set(
            assignments.loc[
                assignments["fold"].astype(int).ne(fold),
                "cohort_visit_number",
            ].astype(int)
        )
        recomputed_domains = {}
        for domain in domains:
            observed_concepts = concepts.loc[
                concepts["fold"].eq(fold) & concepts["domain"].eq(domain)
            ].copy()
            recomputed_selection = select_concepts(
                events[domain],
                training_visits,
                domain,
                fold,
                config,
            )
            expected_concepts = recomputed_selection.selected.copy()
            expected_concepts["selection_hash"] = recomputed_selection.selection_hash
            if len(observed_concepts) != int(config["features"]["concept_count"]):
                raise IntegrityError(
                    f"{domain} fold {fold} does not contain exactly 50 selected concepts"
                )
            observed_concepts["selected"] = _strict_boolean(
                observed_concepts["selected"], "concept selected"
            )
            _assert_selection_table_equal(
                observed_concepts,
                expected_concepts,
                [
                    "fold",
                    "domain",
                    "rank",
                    "concept_key",
                    "concept_name",
                    "training_visit_prevalence",
                    "source_table",
                    "semantics",
                    "units",
                    "selected",
                    "selection_hash",
                ],
                f"{domain} fold {fold} concept selection",
            )

            observed_derived = derived.loc[
                derived["fold"].eq(fold) & derived["domain"].eq(domain)
            ].copy()
            recomputed_feature = build_fold_domain_features(
                cohort,
                recomputed_selection,
                events[domain],
                training_visits,
                config,
            )
            recomputed_domains[domain] = recomputed_feature
            expected_candidates = int(
                config["features"][domain]["constructed_count"]
            )
            if len(observed_derived) != expected_candidates:
                raise IntegrityError(
                    f"{domain} fold {fold} has {len(observed_derived)} candidate rows; "
                    f"expected {expected_candidates}"
                )
            observed_derived["selected"] = _strict_boolean(
                observed_derived["selected"], "derived-feature selected"
            )
            _assert_selection_table_equal(
                observed_derived,
                recomputed_feature.selection_audit,
                [
                    "fold",
                    "domain",
                    "rank",
                    "candidate_feature_name",
                    "source_concept",
                    "summary_type",
                    "training_support_count",
                    "training_support_proportion",
                    "selection_score",
                    "tie_break_value",
                    "training_visit_count",
                    "support_definition",
                    "selected",
                    "selection_rule_identifier",
                    "selection_rule_version",
                    "derived_selection_hash",
                ],
                f"{domain} fold {fold} derived-feature selection",
            )
            selected = observed_derived.loc[
                observed_derived["selected"]
            ].sort_values("rank", kind="stable")
            if len(selected) != 21:
                raise IntegrityError(
                    f"{domain} fold {fold} does not select exactly 21 derived columns"
                )
            feature_path = (
                restricted_root
                / "fold_features"
                / f"fold_{fold}"
                / f"{domain}.parquet"
            )
            if not feature_path.is_file():
                raise IntegrityError(f"restricted fold feature table is missing: {feature_path.name}")
            actual_feature = pd.read_parquet(feature_path)
            try:
                pd.testing.assert_frame_equal(
                    actual_feature.reset_index(drop=True),
                    recomputed_feature.frame.reset_index(drop=True),
                    check_dtype=True,
                    check_exact=True,
                )
            except AssertionError as error:
                raise IntegrityError(
                    f"{domain} fold {fold} feature table differs from recomputed "
                    "training-only construction"
                ) from error

        if not matrix_manifest.empty:
            for matrix_name in config["matrices"]:
                matrix = assemble_matrix(
                    baseline,
                    recomputed_domains,
                    matrix_name,
                    config,
                )
                observed_matrix = matrix_manifest.loc[
                    matrix_manifest["fold"].astype(int).eq(fold)
                    & matrix_manifest["matrix"].eq(matrix_name)
                ]
                if len(observed_matrix) != 1:
                    raise IntegrityError(
                        f"{matrix_name} fold {fold} matrix manifest row is missing or duplicated"
                    )
                row = observed_matrix.iloc[0]
                expected_schema_hash = hash_frame_schema(matrix)
                expected_value_hash = hash_frame_values(
                    matrix.reset_index(),
                    identity_columns=["cohort_visit_number"],
                )
                if (
                    int(row["rows"]) != len(matrix)
                    or int(row["input_feature_count"]) != matrix.shape[1]
                    or row["feature_schema_hash"] != expected_schema_hash
                    or row["feature_matrix_hash"] != expected_value_hash
                ):
                    raise IntegrityError(
                        f"{matrix_name} fold {fold} matrix manifest differs from "
                        "the recomputed selected domain columns"
                    )


def _assert_selection_table_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    """Compare evidence in configured rank order, including recomputed hashes."""
    observed_ordered = observed.sort_values("rank", kind="stable").reset_index(drop=True)
    expected_ordered = expected.sort_values("rank", kind="stable").reset_index(drop=True)
    if observed_ordered["rank"].tolist() != list(range(1, len(observed_ordered) + 1)):
        raise IntegrityError(f"{label} ranks are not consecutive and unique")
    for column in columns:
        left = observed_ordered[column]
        right = expected_ordered[column]
        if pd.api.types.is_numeric_dtype(right) and not pd.api.types.is_bool_dtype(
            right
        ):
            equal = np.allclose(
                pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
                rtol=0,
                atol=5e-11,
                equal_nan=True,
            )
        else:
            equal = left.astype("string").fillna("<NA>").equals(
                right.astype("string").fillna("<NA>")
            )
        if not equal:
            raise IntegrityError(f"{label} has invalid {column}")


def _strict_boolean(values: pd.Series, label: str) -> pd.Series:
    """Parse manifest booleans without treating arbitrary strings as true."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.casefold()
    unknown = normalized.notna() & ~normalized.isin({"true", "false", "1", "0"})
    if unknown.any():
        raise IntegrityError(f"{label} contains invalid boolean values")
    return normalized.map({"true": True, "1": True, "false": False, "0": False}).fillna(
        False
    )


def _assert_count_comparison_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    label: str,
) -> None:
    columns = [
        *keys,
        "expected",
        "observed",
        "absolute_difference",
        "tolerance",
        "matches",
    ]
    if set(columns) - set(observed):
        raise IntegrityError(f"{label} is missing required evidence columns")
    left = observed[columns].sort_values(keys, kind="stable").reset_index(drop=True)
    right = expected[columns].sort_values(keys, kind="stable").reset_index(drop=True)
    left["matches"] = _strict_boolean(left["matches"], f"{label} matches")
    right["matches"] = _strict_boolean(right["matches"], f"{label} matches")
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise IntegrityError(
            f"{label} differs from configuration and observed audit counts"
        ) from error


def _verify_dataset_invariants(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    expected = {
        "folds": 5,
        "matrices": 8,
        "models": 4,
        "model_fits": 160,
        "oof_probabilities_per_visit": 32,
    }
    mismatches = {
        key: (expected_value, manifest.get(key))
        for key, expected_value in expected.items()
        if manifest.get(key) != expected_value
    }
    if mismatches:
        raise IntegrityError(f"Dataset manifest invariant mismatch: {mismatches}")
    required = {
        "fold_metrics.csv",
        "pooled_oof_metrics.csv",
        "selected_model_performance_table.csv",
        "selected_model_clinical_utility_table.csv",
        "selected_model_calibration_table.csv",
        "selected_models_roc_coordinates.csv",
        "selected_models_decision_curve_coordinates.csv",
        "prespecified_paired_matrix_comparisons.csv",
    }
    missing = [name for name in sorted(required) if not (dataset_dir / name).is_file()]
    if missing:
        raise IntegrityError(f"Required analytical outputs are missing: {missing}")


def _verify_output_schemas(dataset_dir: Path) -> None:
    schemas = read_yaml(PROJECT_ROOT / "mappings" / "schemas" / "output_tables.yaml")
    filenames = {
        "fold_metrics": "fold_metrics.csv",
        "pooled_oof_metrics": "pooled_oof_metrics.csv",
        "confidence_intervals": "all_metric_confidence_intervals.csv",
        "roc_coordinates": "selected_models_roc_coordinates.csv",
        "calibration_coordinates": "selected_models_calibration_coordinates.csv",
        "decision_curve_coordinates": "selected_models_decision_curve_coordinates.csv",
    }
    for schema_name, filename in filenames.items():
        frame = pd.read_csv(dataset_dir / filename)
        definition = schemas[schema_name]
        required = set(definition["keys"]) | set(definition["required"])
        missing = required - set(frame.columns)
        if missing:
            raise IntegrityError(f"{filename} is missing schema columns: {sorted(missing)}")
        if frame.duplicated(definition["keys"], keep=False).any():
            raise IntegrityError(f"{filename} contains duplicate schema keys")


def _public_output_hashes(public_dir: Path) -> dict[str, str]:
    excluded = {
        "run_manifest.json",
        "manifests/output_manifest.json",
    }
    return {
        path.relative_to(public_dir).as_posix(): hash_file(path)
        for path in sorted(public_dir.rglob("*"))
        if path.is_file() and path.relative_to(public_dir).as_posix() not in excluded
    }


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "shap": shap.__version__,
    }


def _code_hash() -> str:
    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
    ]
    files = [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    files.append(PROJECT_ROOT / "run_pipeline.py")
    return hash_object(
        {
            path.relative_to(PROJECT_ROOT).as_posix(): hash_file(path)
            for path in sorted(files)
        }
    )


def _partial_result(
    config: dict[str, Any],
    public_dir: Path,
    restricted_dir: Path,
    standardized: Any,
) -> RunResult:
    manifest = {
        "dataset": config["dataset"],
        "partial": True,
        "config_hash": config["_meta"]["config_hash"],
        "mapping_hash": standardized.mapping_hash,
        "created_utc": utc_timestamp(),
    }
    write_json(manifest, public_dir / "run_manifest.json")
    return RunResult(config["dataset"], public_dir, restricted_dir, manifest)
