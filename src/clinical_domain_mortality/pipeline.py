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
import sklearn

from .adapters import CHoRUSAdapter, MIMICIVAdapter, SourceAdapter
from .audit import git_commit, git_is_dirty, scan_public_tree, utc_timestamp
from .cohort import build_cohort, create_patient_folds
from .config import PROJECT_ROOT, load_config, read_yaml, resolve_project_path
from .errors import ConfigurationError, IntegrityError
from .evaluation import evaluate_predictions
from .features import (
    assemble_matrix,
    build_fold_domain_features,
    prepare_domain_events,
    select_concepts,
)
from .hashing import hash_file, hash_object
from .io import read_json, verify_hashes, write_csv, write_json
from .modeling import fit_predict_fold, validate_oof_predictions


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
    public_dir = public_base / dataset
    restricted_dir = private_base / dataset
    if public_dir.exists():
        shutil.rmtree(public_dir)
    if restricted_dir.exists():
        shutil.rmtree(restricted_dir)
    public_dir.mkdir(parents=True)
    restricted_dir.mkdir(parents=True)
    (public_dir / "manifests").mkdir()

    # Stage 1: validate and normalize source.
    standardized = adapter_for(config).load()
    write_json(
        {
            "mapping_confirmed": True,
            "mapping_hash": standardized.mapping_hash,
            "adapter": config["adapter"],
            "table_columns": {
                name: sorted(frame.columns.tolist())
                for name, frame in standardized.tables.items()
            },
            "row_counts": {
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
    cohort_result = build_cohort(standardized, config)
    cohort_result.cohort.to_parquet(
        restricted_dir / "base_acute_care_cohort.parquet", index=False
    )
    cohort_result.baseline.to_parquet(restricted_dir / "baseline_X.parquet", index=False)
    write_csv(cohort_result.attrition, public_dir / "attrition.csv")
    write_json(
        {
            "cohort_hash": cohort_result.cohort_hash,
            "row_order_hash": cohort_result.row_order_hash,
            "visits": len(cohort_result.cohort),
            "patients": int(cohort_result.cohort["patient_id"].nunique()),
            "outcomes": int(cohort_result.cohort["outcome"].sum()),
            "landmark_hours": config["cohort"]["landmark_hours"],
            "outcome_horizon_days": config["cohort"]["outcome_horizon_days"],
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
    prepared = prepare_domain_events(standardized, cohort_result.cohort, config)
    write_csv(prepared.audit, public_dir / "event_linkage_audit.csv")
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
            if not selection.unit_audit.empty:
                audit = selection.unit_audit.copy()
                audit.insert(0, "fold", fold)
                unit_audits.append(audit)
            feature_manifest_rows.append(
                {
                    "fold": fold,
                    "domain": domain,
                    "feature_count": len(feature.feature_names),
                    "feature_hash": feature.feature_hash,
                    "selection_hash": selection.selection_hash,
                }
            )
            feature_dictionary_rows.extend(
                {
                    "fold": fold,
                    "domain": domain,
                    "feature_name": feature_name,
                    "selection_hash": selection.selection_hash,
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
                    "feature_names_hash": hash_object(matrix.columns.tolist()),
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
                    **fit.manifest,
                }
                fit_manifests.append(manifest)
                importance = fit.feature_importance.copy()
                importance.insert(0, "fold", fold)
                importance.insert(1, "matrix", matrix_name)
                importance_parts.append(importance)

    selections = pd.concat(selection_parts, ignore_index=True)
    selection_target = (
        public_dir / "fold_concept_selections.csv"
        if config.get("synthetic")
        else restricted_dir / "fold_concept_selections.csv"
    )
    write_csv(selections, selection_target)
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
    if unit_audits:
        write_csv(pd.concat(unit_audits, ignore_index=True), public_dir / "measurement_unit_audit.csv")
    else:
        write_csv(
            pd.DataFrame(columns=["fold", "concept_key", "unit", "status", "count"]),
            public_dir / "measurement_unit_audit.csv",
        )
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
            "training_folds_only": True,
        },
        public_dir / "manifests" / "selection_manifest.json",
    )
    if stop_after == 6:
        return _partial_result(config, public_dir, restricted_dir, standardized)

    predictions = pd.concat(predictions_parts, ignore_index=True)
    validate_oof_predictions(predictions, cohort_result.cohort, matrix_names, model_names)
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
    write_json(
        {
            "dataset": dataset,
            "adapter": config["adapter"],
            "input_hashes": standardized.input_hashes,
            "input_collection_hash": hash_object(standardized.input_hashes),
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
            "bootstrap_repetitions": evaluation_result.bootstrap_repetitions,
        },
        public_dir / "manifests" / "dataset_manifest.json",
    )
    software = _software_versions()
    code_hash = _code_hash()
    output_hashes = _public_output_hashes(public_dir)
    write_json(output_hashes, public_dir / "manifests" / "output_manifest.json")
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
        "restricted_artifacts": [
            "base_acute_care_cohort.parquet",
            "baseline_X.parquet",
            "fold_assignments_restricted.csv",
            "prepared_measurements.parquet",
            "prepared_medications.parquet",
            "prepared_procedures.parquet",
            "fold_features/fold_<k>/<domain>.parquet",
            "oof_predictions_restricted.csv",
        ],
    }
    write_json(run_manifest, public_dir / "run_manifest.json")
    scan_public_tree(public_dir)
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
        dataset_manifest = read_json(dataset_dir / "manifests" / "dataset_manifest.json")
        _verify_dataset_invariants(dataset_dir, dataset_manifest)
        _verify_output_schemas(dataset_dir)
        verified[dataset] = {
            "run_id": manifest["run_id"],
            "files_verified": len(manifest["output_hashes"]),
        }
    if set(datasets) == {"chorus", "mimiciv"}:
        expected = read_json(PROJECT_ROOT / "synthetic_data" / "expected_outputs" / "expected_summary.json")
        expected_hashes = read_json(
            PROJECT_ROOT
            / "synthetic_data"
            / "expected_outputs"
            / "expected_aggregate_hashes.json"
        )
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
        for dataset in ("chorus", "mimiciv"):
            verify_hashes(root / dataset, expected_hashes[dataset])
        actual_summary_hash = hash_file(root / "synthetic_run_summary.csv")
        if actual_summary_hash != expected_hashes["synthetic_run_summary.csv"]:
            raise IntegrityError("Synthetic aggregate summary hash does not match the release")
    return {"verified": verified, "status": "ok"}


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
