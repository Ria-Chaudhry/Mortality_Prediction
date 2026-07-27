"""Complete pooled-OOF analytical output construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..errors import IntegrityError
from .bootstrap import bootstrap_indices, percentile_interval
from .metrics import (
    binary_metrics,
    calibration_summary,
    net_benefit,
    roc_coordinates,
    sensitivity_at_specificity,
    top_risk_analysis,
)


@dataclass
class EvaluationResult:
    tables: dict[str, pd.DataFrame]
    bootstrap_repetitions: int


def evaluate_predictions(
    predictions: pd.DataFrame,
    cohort: pd.DataFrame,
    config: dict[str, Any],
) -> EvaluationResult:
    """Generate every aggregate table directly from held-out probabilities."""
    evaluation = config["evaluation"]
    threshold = float(evaluation["classification_threshold"])
    identity = cohort[
        ["cohort_visit_number", "patient_id", "outcome"]
    ].copy()
    merged = predictions.merge(
        identity, on="cohort_visit_number", how="left", validate="many_to_one"
    )
    if merged[["patient_id", "outcome"]].isna().any().any():
        raise IntegrityError("OOF predictions do not reconcile one-to-one with the cohort")
    fold_metrics = _fold_metrics(merged, threshold)
    fold_summary = _fold_summary(fold_metrics)
    pooled = _pooled_metrics(merged, threshold)
    selected = _select_models(pooled, config)

    bootstrap_config = evaluation["bootstrap"]
    base_identity = identity.sort_values("cohort_visit_number", kind="stable").reset_index(drop=True)
    replicates = bootstrap_indices(
        base_identity["patient_id"],
        int(bootstrap_config["repetitions"]),
        int(bootstrap_config["seed"]),
    )
    confidence = _metric_confidence_intervals(
        merged, base_identity, replicates, threshold, float(bootstrap_config["confidence_level"])
    )
    selected_performance = _selected_performance(selected, pooled, confidence)
    selected_outputs = _selected_model_outputs(
        merged, base_identity, selected, replicates, config
    )
    comparisons = _paired_comparisons(
        merged, base_identity, selected, replicates, config
    )
    tables = {
        "fold_metrics.csv": fold_metrics,
        "fold_metric_summaries.csv": fold_summary,
        "pooled_oof_metrics.csv": pooled,
        "best_model_by_matrix.csv": selected,
        "all_metric_confidence_intervals.csv": confidence,
        "selected_model_performance_table.csv": selected_performance,
        "prespecified_paired_matrix_comparisons.csv": comparisons,
        **selected_outputs,
    }
    return EvaluationResult(tables=tables, bootstrap_repetitions=len(replicates))


def _fold_metrics(merged: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (matrix, model, fold), group in merged.groupby(
        ["matrix", "model", "fold"], sort=True
    ):
        row = {
            "matrix": matrix,
            "model": model,
            "fold": int(fold),
            "visits": len(group),
            "events": int(group["outcome"].sum()),
        }
        row.update(binary_metrics(group["outcome"], group["probability"], threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def _fold_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "auroc",
        "auprc",
        "brier",
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "ppv",
        "npv",
        "sensitivity",
        "specificity",
        "f1",
    ]
    rows = []
    for (matrix, model), group in fold_metrics.groupby(["matrix", "model"], sort=True):
        for metric in metric_columns:
            rows.append(
                {
                    "matrix": matrix,
                    "model": model,
                    "metric": metric,
                    "fold_mean": float(group[metric].mean()),
                    "fold_sd": float(group[metric].std(ddof=1)),
                    "valid_folds": int(group[metric].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def _pooled_metrics(merged: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (matrix, model), group in merged.groupby(["matrix", "model"], sort=True):
        row = {
            "matrix": matrix,
            "model": model,
            "visits": len(group),
            "events": int(group["outcome"].sum()),
        }
        row.update(binary_metrics(group["outcome"], group["probability"], threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def _select_models(pooled: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    order = {
        name: index for index, name in enumerate(config["models"]["frozen_order"])
    }
    ranked = pooled.assign(_model_order=pooled["model"].map(order))
    ranked = ranked.sort_values(
        ["matrix", "auprc", "auroc", "brier", "_model_order"],
        ascending=[True, False, False, True, True],
        kind="stable",
    )
    selected = ranked.groupby("matrix", sort=True).head(1).copy()
    selected["selection_rule"] = "AUPRC desc; AUROC desc; Brier asc; frozen model order"
    return selected[
        ["matrix", "model", "auprc", "auroc", "brier", "selection_rule"]
    ].reset_index(drop=True)


def _aligned_combination(
    merged: pd.DataFrame,
    identity: pd.DataFrame,
    matrix: str,
    model: str,
) -> pd.DataFrame:
    selected = merged.loc[
        (merged["matrix"] == matrix) & (merged["model"] == model),
        ["cohort_visit_number", "probability"],
    ]
    result = identity.merge(
        selected, on="cohort_visit_number", how="left", validate="one_to_one"
    )
    if result["probability"].isna().any():
        raise IntegrityError(f"Missing aligned probabilities for {matrix}/{model}")
    return result


def _metric_confidence_intervals(
    merged: pd.DataFrame,
    identity: pd.DataFrame,
    replicates: list[np.ndarray],
    threshold: float,
    confidence_level: float,
) -> pd.DataFrame:
    metric_names = [
        "auroc",
        "auprc",
        "brier",
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "ppv",
        "npv",
        "sensitivity",
        "specificity",
        "f1",
    ]
    rows = []
    combinations = merged[["matrix", "model"]].drop_duplicates().sort_values(
        ["matrix", "model"], kind="stable"
    )
    for combination in combinations.itertuples(index=False):
        aligned = _aligned_combination(
            merged, identity, combination.matrix, combination.model
        )
        y = aligned["outcome"].to_numpy(dtype=int)
        p = aligned["probability"].to_numpy(dtype=float)
        point = binary_metrics(y, p, threshold)
        replicate_metrics = [
            binary_metrics(y[index], p[index], threshold) for index in replicates
        ]
        for metric in metric_names:
            lower, upper, invalid, valid = percentile_interval(
                (item[metric] for item in replicate_metrics), confidence_level
            )
            rows.append(
                {
                    "matrix": combination.matrix,
                    "model": combination.model,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": confidence_level,
                    "bootstrap_repetitions": len(replicates),
                    "valid_replicates": valid,
                    "invalid_replicates": invalid,
                    "bootstrap_unit": "patient",
                    "ci_method": "percentile",
                }
            )
    return pd.DataFrame(rows)


def _selected_performance(
    selected: pd.DataFrame,
    pooled: pd.DataFrame,
    confidence: pd.DataFrame,
) -> pd.DataFrame:
    selected_metrics = selected[["matrix", "model"]].merge(
        pooled, on=["matrix", "model"], validate="one_to_one"
    )
    for metric in ("auroc", "auprc", "brier"):
        interval = confidence.loc[
            confidence["metric"] == metric,
            ["matrix", "model", "ci_lower", "ci_upper"],
        ].rename(
            columns={
                "ci_lower": f"{metric}_ci_lower",
                "ci_upper": f"{metric}_ci_upper",
            }
        )
        selected_metrics = selected_metrics.merge(
            interval, on=["matrix", "model"], how="left", validate="one_to_one"
        )
    return selected_metrics.sort_values("matrix", kind="stable").reset_index(drop=True)


def _selected_model_outputs(
    merged: pd.DataFrame,
    identity: pd.DataFrame,
    selected: pd.DataFrame,
    replicates: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    evaluation = config["evaluation"]
    roc_parts = []
    operating_rows = []
    top_rows = []
    calibration_rows = []
    calibration_parts = []
    decision_parts = []
    for row in selected.itertuples(index=False):
        aligned = _aligned_combination(merged, identity, row.matrix, row.model)
        y = aligned["outcome"]
        p = aligned["probability"]
        roc = roc_coordinates(y, p)
        roc.insert(0, "model", row.model)
        roc.insert(0, "matrix", row.matrix)
        roc_parts.append(roc)

        operating = sensitivity_at_specificity(
            y, p, float(evaluation["specificity_target"])
        )
        operating_rows.append({"matrix": row.matrix, "model": row.model, **operating})
        top = top_risk_analysis(
            aligned["cohort_visit_number"],
            y,
            p,
            float(evaluation["top_risk_fraction"]),
        )
        top_rows.append({"matrix": row.matrix, "model": row.model, **top})

        summary, coordinates = calibration_summary(
            y, p, int(evaluation["calibration_bins"])
        )
        calibration_rows.append({"matrix": row.matrix, "model": row.model, **summary})
        coordinates.insert(0, "model", row.model)
        coordinates.insert(0, "matrix", row.matrix)
        calibration_parts.append(coordinates)
        decision_parts.append(
            _decision_curve(
                aligned,
                row.matrix,
                row.model,
                replicates,
                evaluation,
            )
        )
    operating_frame = pd.DataFrame(operating_rows)
    top_frame = pd.DataFrame(top_rows)
    clinical = operating_frame.merge(
        top_frame,
        on=["matrix", "model"],
        how="inner",
        suffixes=("_at_90_specificity", "_top_10_percent"),
        validate="one_to_one",
    )
    return {
        "selected_models_roc_coordinates.csv": pd.concat(roc_parts, ignore_index=True),
        "selected_models_sensitivity_at_90_specificity.csv": operating_frame,
        "selected_models_top_10_percent_risk_analysis.csv": top_frame,
        "selected_model_clinical_utility_table.csv": clinical,
        "selected_models_calibration_summary.csv": pd.DataFrame(calibration_rows),
        "selected_model_calibration_table.csv": pd.DataFrame(calibration_rows),
        "selected_models_calibration_coordinates.csv": pd.concat(
            calibration_parts, ignore_index=True
        ),
        "selected_models_decision_curve_coordinates.csv": pd.concat(
            decision_parts, ignore_index=True
        ),
    }


def _decision_curve(
    aligned: pd.DataFrame,
    matrix: str,
    model: str,
    replicates: list[np.ndarray],
    evaluation: dict[str, Any],
) -> pd.DataFrame:
    settings = evaluation["decision_thresholds"]
    thresholds = np.round(
        np.arange(
            float(settings["start"]),
            float(settings["stop"]) + float(settings["step"]) / 2,
            float(settings["step"]),
        ),
        10,
    )
    y = aligned["outcome"].to_numpy(dtype=int)
    p = aligned["probability"].to_numpy(dtype=float)
    confidence_level = float(evaluation["bootstrap"]["confidence_level"])
    rows = []
    for threshold in thresholds:
        strategies = {
            "model": p >= threshold,
            "treat_all": np.ones(len(y), dtype=bool),
            "treat_none": np.zeros(len(y), dtype=bool),
        }
        for strategy, flags in strategies.items():
            estimate = net_benefit(y, flags, float(threshold))
            values = [
                net_benefit(y[index], flags[index], float(threshold))
                for index in replicates
            ]
            lower, upper, invalid, valid = percentile_interval(values, confidence_level)
            rows.append(
                {
                    "matrix": matrix,
                    "model": model,
                    "threshold": float(threshold),
                    "strategy": strategy,
                    "net_benefit": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "valid_replicates": valid,
                    "invalid_replicates": invalid,
                }
            )
    return pd.DataFrame(rows)


def _paired_comparisons(
    merged: pd.DataFrame,
    identity: pd.DataFrame,
    selected: pd.DataFrame,
    replicates: list[np.ndarray],
    config: dict[str, Any],
) -> pd.DataFrame:
    selected_models = selected.set_index("matrix")["model"].to_dict()
    confidence_level = float(config["evaluation"]["bootstrap"]["confidence_level"])
    rows = []
    for comparator, expanded in config["evaluation"]["paired_comparisons"]:
        comparator_model = selected_models[comparator]
        expanded_model = selected_models[expanded]
        first = _aligned_combination(merged, identity, comparator, comparator_model)
        second = _aligned_combination(merged, identity, expanded, expanded_model)
        y = first["outcome"].to_numpy(dtype=int)
        p_comparator = first["probability"].to_numpy(dtype=float)
        p_expanded = second["probability"].to_numpy(dtype=float)
        first_point = binary_metrics(y, p_comparator)
        second_point = binary_metrics(y, p_expanded)
        replicate_values = {"auroc": [], "auprc": [], "brier_improvement": []}
        for index in replicates:
            comparator_metrics = binary_metrics(y[index], p_comparator[index])
            expanded_metrics = binary_metrics(y[index], p_expanded[index])
            replicate_values["auroc"].append(
                expanded_metrics["auroc"] - comparator_metrics["auroc"]
            )
            replicate_values["auprc"].append(
                expanded_metrics["auprc"] - comparator_metrics["auprc"]
            )
            replicate_values["brier_improvement"].append(
                comparator_metrics["brier"] - expanded_metrics["brier"]
            )
        points = {
            "auroc": second_point["auroc"] - first_point["auroc"],
            "auprc": second_point["auprc"] - first_point["auprc"],
            "brier_improvement": first_point["brier"] - second_point["brier"],
        }
        for metric, point in points.items():
            lower, upper, invalid, valid = percentile_interval(
                replicate_values[metric], confidence_level
            )
            rows.append(
                {
                    "comparator_matrix": comparator,
                    "comparator_model": comparator_model,
                    "expanded_matrix": expanded,
                    "expanded_model": expanded_model,
                    "metric": metric,
                    "difference": point,
                    "difference_definition": (
                        "comparator minus expanded"
                        if metric == "brier_improvement"
                        else "expanded minus comparator"
                    ),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "valid_replicates": valid,
                    "invalid_replicates": invalid,
                    "shared_patient_bootstrap": True,
                }
            )
    return pd.DataFrame(rows)
