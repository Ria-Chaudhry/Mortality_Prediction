from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clinical_domain_mortality.errors import IntegrityError, LeakageError
from clinical_domain_mortality.features.validation import assert_no_forbidden_features
from clinical_domain_mortality.modeling import fit_predict_fold, validate_oof_predictions


def test_imputation_and_scaling_fit_training_only(chorus_config):
    x_train = pd.DataFrame({"numeric": [1.0, np.nan, 3.0, 5.0], "varying": [0, 1, 0, 1]})
    x_validation = pd.DataFrame({"numeric": [1000.0, np.nan], "varying": [1, 0]})
    y = pd.Series([0, 1, 0, 1])
    fit = fit_predict_fold(x_train, y, x_validation, "logistic_regression", chorus_config)
    imputer = fit.pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
    assert imputer.statistics_[0] == 3.0
    assert fit.manifest["training_only_preprocessing"] is True


@pytest.mark.parametrize(
    "name",
    [
        "patient_id",
        "visit_id",
        "event_datetime",
        "death_indicator",
        "outcome",
        "discharge_disposition",
        "length_of_stay",
        "post_landmark_value",
        "prediction_probability",
    ],
)
def test_forbidden_predictor_names_hard_fail(name, chorus_config):
    with pytest.raises(LeakageError):
        assert_no_forbidden_features(pd.DataFrame({name: [1, 2]}), chorus_config)


def test_datetime_predictor_hard_fails(chorus_config):
    with pytest.raises(LeakageError):
        assert_no_forbidden_features(
            pd.DataFrame({"innocent_name": pd.to_datetime(["2020-01-01"])}),
            chorus_config,
        )


def test_duplicate_or_missing_oof_predictions_hard_fail(cohort_result):
    cohort = cohort_result.cohort.iloc[:2]
    duplicate = pd.DataFrame(
        {
            "cohort_visit_number": [1, 1],
            "matrix": ["baseline", "baseline"],
            "model": ["m", "m"],
            "fold": [0, 0],
            "probability": [0.1, 0.2],
        }
    )
    with pytest.raises(IntegrityError):
        validate_oof_predictions(duplicate, cohort, ["baseline"], ["m"])


def test_invalid_probability_hard_fails(cohort_result):
    cohort = cohort_result.cohort.iloc[:2].copy()
    predictions = pd.DataFrame(
        {
            "cohort_visit_number": cohort["cohort_visit_number"],
            "matrix": "baseline",
            "model": "m",
            "fold": [0, 1],
            "probability": [0.2, 1.2],
        }
    )
    with pytest.raises(IntegrityError):
        validate_oof_predictions(predictions, cohort, ["baseline"], ["m"])


def test_wrong_positive_class_labels_hard_fail(chorus_config):
    x = pd.DataFrame({"x": [0, 1, 2, 3], "z": [1, 0, 1, 0]})
    with pytest.raises(IntegrityError):
        fit_predict_fold(
            x,
            pd.Series([0, 2, 0, 2]),
            x.iloc[:1],
            "logistic_regression",
            chorus_config,
        )


def test_duplicate_feature_names_hard_fail(chorus_config):
    frame = pd.DataFrame([[1, 2]], columns=["feature", "feature"])
    with pytest.raises(IntegrityError):
        assert_no_forbidden_features(frame, chorus_config)


def test_repeated_seeded_fit_is_deterministic(chorus_config):
    x = pd.DataFrame(
        {"x": [0, 1, 2, 3, 4, 5], "category": ["a", "b", "a", "b", "a", "b"]}
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])
    first = fit_predict_fold(x, y, x, "random_forest", chorus_config)
    second = fit_predict_fold(x, y, x, "random_forest", chorus_config)
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
