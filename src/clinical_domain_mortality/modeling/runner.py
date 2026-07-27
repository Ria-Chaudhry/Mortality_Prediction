"""Fold-safe preprocessing, frozen estimators, and positive-class extraction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..errors import IntegrityError


@dataclass
class FitResult:
    probabilities: np.ndarray
    manifest: dict[str, Any]
    feature_importance: pd.DataFrame
    pipeline: Pipeline


def fit_predict_fold(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    model_name: str,
    config: dict[str, Any],
) -> FitResult:
    """Fit every learned operation on training rows and predict validation only."""
    if y_train.nunique() != 2 or set(y_train.astype(int).unique()) != {0, 1}:
        raise IntegrityError("Training labels must contain exactly classes 0 and 1")
    if list(x_train.columns) != list(x_validation.columns):
        raise IntegrityError("Training and validation feature definitions differ within a fold")
    pipeline = build_pipeline(x_train, model_name, config)
    pipeline.fit(x_train, y_train.astype(int))
    with warnings.catch_warnings():
        if model_name == "lightgbm":
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, but LGBMClassifier "
                    "was fitted with feature names"
                ),
                category=UserWarning,
            )
        probability_matrix = pipeline.predict_proba(x_validation)
    classes = np.asarray(pipeline.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if len(positive_positions) != 1:
        raise IntegrityError(f"Cannot identify unique positive class from {classes.tolist()}")
    probabilities = np.asarray(probability_matrix[:, positive_positions[0]], dtype=float)
    if probabilities.shape != (len(x_validation),):
        raise IntegrityError("Positive-class probability extraction has invalid shape")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise IntegrityError("Estimator returned missing or out-of-range probabilities")
    selector = pipeline.named_steps["variance"]
    importance = _feature_importance(pipeline, model_name)
    manifest = {
        "model": model_name,
        "training_rows": len(x_train),
        "validation_rows": len(x_validation),
        "input_feature_count": int(x_train.shape[1]),
        "retained_transformed_feature_count": int(selector.get_support().sum()),
        "positive_class": 1,
        "classes": classes.astype(int).tolist(),
        "training_only_preprocessing": True,
        "hyperparameters": _configured_hyperparameters(model_name, config),
    }
    return FitResult(probabilities, manifest, importance, pipeline)


def build_pipeline(
    frame: pd.DataFrame, model_name: str, config: dict[str, Any]
) -> Pipeline:
    categorical = [
        column
        for column in frame.columns
        if isinstance(frame[column].dtype, pd.StringDtype)
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
        or frame[column].dtype == object
    ]
    numeric = [column for column in frame.columns if column not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    preprocessing = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    steps: list[tuple[str, Any]] = [
        ("preprocessing", preprocessing),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]
    if model_name == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", _estimator(model_name, config)))
    return Pipeline(steps)


def _estimator(model_name: str, config: dict[str, Any]) -> Any:
    settings = config["models"][model_name]
    if model_name == "logistic_regression":
        return LogisticRegression(**settings)
    if model_name == "random_forest":
        return RandomForestClassifier(**settings)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(**settings)
    if model_name == "lightgbm":
        return LGBMClassifier(**settings)
    raise IntegrityError(f"Unknown frozen model: {model_name}")


def _configured_hyperparameters(
    model_name: str, config: dict[str, Any]
) -> dict[str, Any]:
    return dict(config["models"][model_name])


def _feature_importance(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    names = names[pipeline.named_steps["variance"].get_support()]
    estimator = pipeline.named_steps["estimator"]
    if hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_)[0])
        kind = "absolute_coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "impurity_or_gain"
    else:
        values = np.full(len(names), np.nan)
        kind = "unavailable"
    return pd.DataFrame(
        {
            "model": model_name,
            "transformed_feature": names.astype(str),
            "importance": values,
            "importance_definition": kind,
        }
    ).sort_values(["importance", "transformed_feature"], ascending=[False, True], kind="stable")


def validate_oof_predictions(
    predictions: pd.DataFrame,
    cohort: pd.DataFrame,
    matrix_names: list[str],
    model_names: list[str],
) -> None:
    """Hard fail unless every visit has exactly one valid OOF value per combination."""
    required = {"cohort_visit_number", "matrix", "model", "fold", "probability"}
    missing = required - set(predictions)
    if missing:
        raise IntegrityError(f"OOF predictions missing columns: {sorted(missing)}")
    expected_count = len(cohort) * len(matrix_names) * len(model_names)
    if len(predictions) != expected_count:
        raise IntegrityError(
            f"Expected {expected_count} OOF rows, observed {len(predictions)}"
        )
    key = ["cohort_visit_number", "matrix", "model"]
    if predictions.duplicated(key, keep=False).any():
        raise IntegrityError("A visit has multiple OOF probabilities for a model-matrix combination")
    expected_visits = set(cohort["cohort_visit_number"])
    for matrix in matrix_names:
        for model in model_names:
            actual = set(
                predictions.loc[
                    (predictions["matrix"] == matrix) & (predictions["model"] == model),
                    "cohort_visit_number",
                ]
            )
            if actual != expected_visits:
                raise IntegrityError(f"Missing OOF predictions for {matrix}/{model}")
    probability = predictions["probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise IntegrityError("OOF probabilities are missing or outside [0, 1]")
