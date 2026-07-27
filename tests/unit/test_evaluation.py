from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from clinical_domain_mortality.evaluation import bootstrap_indices, evaluate_predictions
from clinical_domain_mortality.evaluation.bootstrap import percentile_interval
from clinical_domain_mortality.evaluation.metrics import (
    calibration_summary,
    net_benefit,
    sensitivity_at_specificity,
    top_risk_analysis,
)


def test_top_ten_percent_resolves_boundary_ties_by_row_number():
    result = top_risk_analysis(
        pd.Series(range(1, 11)),
        pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        pd.Series([0.5] * 10),
        0.10,
    )
    assert result["flagged_count"] == 1
    assert result["cutoff_ties_total"] == 10
    assert result["cutoff_ties_flagged"] == 1
    assert result["deaths_captured"] == 1


def test_calibration_bins_are_deterministic_with_ties():
    summary, coordinates = calibration_summary(
        pd.Series([0, 1] * 5), pd.Series([0.2] * 10), 10
    )
    assert summary["actual_bin_count"] == 10
    assert coordinates["count"].sum() == 10
    assert np.isfinite(summary["expected_calibration_error"])


def test_sensitivity_at_ninety_specificity_tie_rule():
    result = sensitivity_at_specificity(
        pd.Series([0, 0, 0, 0, 1, 1]),
        pd.Series([0.1, 0.2, 0.3, 0.4, 0.9, 0.8]),
        0.90,
    )
    assert result["sensitivity"] == 1
    assert result["specificity"] >= 0.90


def test_decision_curve_formula():
    y = np.array([1, 0, 1, 0])
    flags = np.array([True, True, False, False])
    assert net_benefit(y, flags, 0.2) == 0.25 - 0.25 * 0.2 / 0.8


def test_patient_bootstrap_keeps_cluster_visits_together():
    patients = pd.Series(["a", "a", "b", "c", "c", "c"])
    samples = bootstrap_indices(patients, 20, 9)
    for sample in samples:
        count_a = int(np.isin(sample, [0, 1]).sum())
        count_c = int(np.isin(sample, [3, 4, 5]).sum())
        assert count_a % 2 == 0
        assert count_c % 3 == 0


def test_rare_outcome_invalid_bootstrap_replicates_are_reported():
    lower, upper, invalid, valid = percentile_interval(
        [0.5, float("nan"), 0.7, float("nan")], 0.95
    )
    assert invalid == 2
    assert valid == 2
    assert lower <= upper


def test_complete_evaluation_and_paired_bootstrap_outputs(chorus_config):
    config = deepcopy(chorus_config)
    config["evaluation"]["bootstrap"]["repetitions"] = 10
    n = 20
    cohort = pd.DataFrame(
        {
            "cohort_visit_number": range(1, n + 1),
            "patient_id": [f"p{i // 2}" for i in range(n)],
            "outcome": [0, 1] * 10,
        }
    )
    rows = []
    for matrix_index, matrix in enumerate(config["matrices"]):
        for model_index, model in enumerate(config["models"]["frozen_order"]):
            probability = np.clip(
                np.linspace(0.05, 0.95, n)
                + matrix_index * 0.001
                + model_index * 0.0001,
                0,
                1,
            )
            for index in range(n):
                rows.append(
                    {
                        "cohort_visit_number": index + 1,
                        "matrix": matrix,
                        "model": model,
                        "fold": index % 5,
                        "probability": probability[index],
                    }
                )
    result = evaluate_predictions(pd.DataFrame(rows), cohort, config)
    assert len(result.tables["best_model_by_matrix.csv"]) == 8
    comparisons = result.tables["prespecified_paired_matrix_comparisons.csv"]
    assert len(comparisons) == 36
    assert comparisons["shared_patient_bootstrap"].all()
    assert set(comparisons["metric"]) == {"auroc", "auprc", "brier_improvement"}
