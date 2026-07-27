"""Exact binary performance and operating-point calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both = np.unique(y).size == 2
    return {
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "auprc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": _ratio(tp + tn, len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else float("nan"),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def roc_coordinates(y_true: pd.Series, probability: pd.Series) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true.astype(int), probability.astype(float), drop_intermediate=False)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "specificity": 1 - fpr,
            "sensitivity": tpr,
        }
    )


def sensitivity_at_specificity(
    y_true: pd.Series, probability: pd.Series, target: float
) -> dict[str, float | int]:
    coordinates = roc_coordinates(y_true, probability)
    eligible = coordinates.loc[coordinates["specificity"] >= target].copy()
    eligible["_distance"] = (eligible["specificity"] - target).abs()
    chosen = eligible.sort_values(
        ["sensitivity", "_distance", "threshold"],
        ascending=[False, True, False],
        kind="stable",
    ).iloc[0]
    threshold = float(chosen["threshold"])
    flagged = probability.to_numpy(dtype=float) >= threshold
    y = y_true.to_numpy(dtype=int)
    tp = int(y[flagged].sum())
    fp = int(flagged.sum() - tp)
    return {
        "specificity_target": float(target),
        "sensitivity": float(chosen["sensitivity"]),
        "specificity": float(chosen["specificity"]),
        "threshold": threshold,
        "ppv": _ratio(tp, tp + fp),
        "flagged_count": int(flagged.sum()),
        "flagged_per_100": 100 * float(flagged.mean()),
    }


def top_risk_analysis(
    cohort_visit_number: pd.Series,
    y_true: pd.Series,
    probability: pd.Series,
    fraction: float,
) -> dict[str, float | int]:
    frame = pd.DataFrame(
        {
            "cohort_visit_number": cohort_visit_number.astype(int),
            "outcome": y_true.astype(int),
            "probability": probability.astype(float),
        }
    ).sort_values(
        ["probability", "cohort_visit_number"],
        ascending=[False, True],
        kind="stable",
    )
    flagged_count = math.ceil(fraction * len(frame))
    flagged = frame.iloc[:flagged_count]
    cutoff = float(flagged.iloc[-1]["probability"])
    cutoff_ties_total = int((frame["probability"] == cutoff).sum())
    cutoff_ties_flagged = int((flagged["probability"] == cutoff).sum())
    deaths = int(flagged["outcome"].sum())
    ppv = _ratio(deaths, flagged_count)
    prevalence = float(frame["outcome"].mean())
    return {
        "requested_fraction": float(fraction),
        "flagged_count": flagged_count,
        "flagged_per_100": 100 * flagged_count / len(frame),
        "deaths_captured": deaths,
        "total_deaths": int(frame["outcome"].sum()),
        "death_capture_fraction": _ratio(deaths, int(frame["outcome"].sum())),
        "ppv": ppv,
        "prevalence": prevalence,
        "enrichment": _ratio(ppv, prevalence),
        "cutoff_probability": cutoff,
        "cutoff_ties_total": cutoff_ties_total,
        "cutoff_ties_flagged": cutoff_ties_flagged,
    }


def calibration_summary(
    y_true: pd.Series,
    probability: pd.Series,
    bins: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    y = y_true.to_numpy(dtype=int)
    p = probability.to_numpy(dtype=float)
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if np.unique(y).size == 2:
        calibration_model = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=2000
        ).fit(logits, y)
        intercept = float(calibration_model.intercept_[0])
        slope = float(calibration_model.coef_[0, 0])
    else:
        intercept = float("nan")
        slope = float("nan")
    prevalence = float(y.mean())
    mean_prediction = float(p.mean())
    calibration_in_the_large = _logit(prevalence) - _logit(mean_prediction)
    coordinates = calibration_coordinates(y_true, probability, bins)
    ece = float(
        (
            coordinates["count"]
            / coordinates["count"].sum()
            * (coordinates["mean_predicted_risk"] - coordinates["observed_event_rate"]).abs()
        ).sum()
    )
    summary = {
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_in_the_large": calibration_in_the_large,
        "mean_predicted_risk": mean_prediction,
        "observed_prevalence": prevalence,
        "expected_calibration_error": ece,
        "requested_bin_count": int(bins),
        "actual_bin_count": len(coordinates),
    }
    return summary, coordinates


def calibration_coordinates(
    y_true: pd.Series, probability: pd.Series, bins: int
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "outcome": y_true.astype(int).to_numpy(),
            "probability": probability.astype(float).to_numpy(),
            "row_order": np.arange(len(y_true)),
        }
    ).sort_values(["probability", "row_order"], kind="stable")
    actual_bins = min(int(bins), len(frame))
    frame["bin"] = np.floor(np.arange(len(frame)) * actual_bins / len(frame)).astype(int) + 1
    grouped = (
        frame.groupby("bin", sort=True)
        .agg(
            count=("outcome", "size"),
            events=("outcome", "sum"),
            mean_predicted_risk=("probability", "mean"),
            observed_event_rate=("outcome", "mean"),
            minimum_probability=("probability", "min"),
            maximum_probability=("probability", "max"),
        )
        .reset_index()
    )
    intervals = [
        _wilson_interval(int(row.events), int(row.count)) for row in grouped.itertuples()
    ]
    grouped["event_rate_ci_lower"] = [item[0] for item in intervals]
    grouped["event_rate_ci_upper"] = [item[1] for item in intervals]
    return grouped


def net_benefit(y_true: np.ndarray, predicted_positive: np.ndarray, threshold: float) -> float:
    y = np.asarray(y_true, dtype=int)
    flag = np.asarray(predicted_positive, dtype=bool)
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    return tp / len(y) - fp / len(y) * threshold / (1 - threshold)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-12, 1 - 1e-12))
    return float(np.log(clipped / (1 - clipped)))


def _wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = events / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)
