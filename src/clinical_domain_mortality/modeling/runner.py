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
from ..hashing import hash_object


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
        "preprocessing_state_hash": _preprocessing_state_hash(pipeline),
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


def _preprocessing_state_hash(pipeline: Pipeline) -> str:
    preprocessing = pipeline.named_steps["preprocessing"]
    state: dict[str, Any] = {"transformers": []}
    for name, transformer, columns in preprocessing.transformers_:
        if name == "remainder":
            continue
        item: dict[str, Any] = {
            "name": name,
            "columns": [str(column) for column in columns],
        }
        if name == "numeric":
            item["imputer_statistics"] = _manifest_values(
                transformer.statistics_
            )
        elif name == "categorical":
            imputer = transformer.named_steps["imputer"]
            encoder = transformer.named_steps["one_hot"]
            item["imputer_statistics"] = _manifest_values(
                imputer.statistics_
            )
            item["encoder_categories"] = [
                _manifest_values(values) for values in encoder.categories_
            ]
        state["transformers"].append(item)
    state["variance_support"] = (
        pipeline.named_steps["variance"].get_support().astype(bool).tolist()
    )
    scaler = pipeline.named_steps.get("scaler")
    if scaler is not None:
        state["scaler_mean"] = _manifest_values(scaler.mean_)
        state["scaler_scale"] = _manifest_values(scaler.scale_)
    return hash_object(state)


def _manifest_values(values: Any) -> list[Any]:
    result: list[Any] = []
    for value in np.asarray(values, dtype=object).tolist():
        if pd.isna(value):
            result.append(None)
        elif isinstance(value, float | np.floating):
            numeric = float(value)
            result.append(numeric if np.isfinite(numeric) else str(numeric))
        elif isinstance(value, int | np.integer):
            result.append(int(value))
        elif isinstance(value, bool | np.bool_):
            result.append(bool(value))
        else:
            result.append(str(value))
    return result


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
    assignments: pd.DataFrame,
    matrix_names: list[str],
    model_names: list[str],
    fit_manifests: list[dict[str, Any]] | None = None,
) -> None:
    """Validate coverage and each row's frozen patient validation-fold identity."""
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
    recorded = predictions.merge(
        assignments[["cohort_visit_number", "fold"]].rename(
            columns={"fold": "assigned_fold"}
        ),
        on="cohort_visit_number",
        how="left",
        validate="many_to_one",
    )
    if recorded["assigned_fold"].isna().any():
        raise IntegrityError("OOF predictions contain unexpected encounters")
    if not recorded["fold"].astype(int).eq(recorded["assigned_fold"].astype(int)).all():
        raise IntegrityError(
            "An OOF prediction's recorded fold does not equal the patient's frozen fold"
        )
    if fit_manifests is not None:
        expected_fits = len(matrix_names) * len(model_names) * assignments["fold"].nunique()
        if len(fit_manifests) != expected_fits:
            raise IntegrityError(
                f"Expected {expected_fits} fit manifests, observed {len(fit_manifests)}"
            )
        keys = [
            (int(item["fold"]), str(item["matrix"]), str(item["model"]))
            for item in fit_manifests
        ]
        if len(keys) != len(set(keys)):
            raise IntegrityError("Duplicate outer-fold fit manifests")
        indexed_assignments = assignments.set_index("cohort_visit_number")
        for item in fit_manifests:
            fold = int(item["fold"])
            training = sorted(
                indexed_assignments.index[
                    indexed_assignments["fold"].astype(int).ne(fold)
                ].astype(int)
            )
            validation = indexed_assignments.index[
                indexed_assignments["fold"].astype(int).eq(fold)
            ].astype(int).tolist()
            expected_training_hash = hash_object(training)
            expected_validation_hash = hash_object(validation)
            if (
                item.get("training_visit_hash") != expected_training_hash
                or item.get("preprocessing_fit_partition_hash")
                != expected_training_hash
                or item.get("validation_visit_hash") != expected_validation_hash
            ):
                raise IntegrityError(
                    "An outer-fold fit manifest does not match the frozen "
                    "training/validation partition"
                )
            state_hash = item.get("preprocessing_state_hash")
            if (
                not isinstance(state_hash, str)
                or len(state_hash) != 64
                or any(character not in "0123456789abcdef" for character in state_hash)
            ):
                raise IntegrityError(
                    "An outer-fold fit manifest lacks a valid preprocessing state hash"
                )
            training_patients = set(
                indexed_assignments.loc[training, "patient_id"].astype(str)
            )
            validation_patients = set(
                indexed_assignments.loc[validation, "patient_id"].astype(str)
            )
            if training_patients & validation_patients:
                raise IntegrityError(
                    "A patient appears in training and validation within an outer fold"
                )
    assignment_required = {"cohort_visit_number", "patient_id", "fold"}
    assignment_missing = assignment_required - set(assignments)
    if assignment_missing:
        raise IntegrityError(
            f"Frozen fold assignments missing columns: {sorted(assignment_missing)}"
        )
    if assignments["cohort_visit_number"].duplicated().any():
        raise IntegrityError("Frozen fold assignments contain duplicate visits")
    cohort_identity = cohort[
        ["cohort_visit_number", "patient_id"]
    ].astype({"patient_id": "string"})
    assignment_identity = assignments[
        ["cohort_visit_number", "patient_id"]
    ].astype({"patient_id": "string"})
    identity = cohort_identity.merge(
        assignment_identity,
        on="cohort_visit_number",
        how="outer",
        suffixes=("_cohort", "_assignment"),
        indicator=True,
    )
    if (identity["_merge"] != "both").any() or (
        identity["patient_id_cohort"] != identity["patient_id_assignment"]
    ).any():
        raise IntegrityError("Frozen fold assignments do not match cohort visit-patient identity")
    if (assignments.groupby("patient_id")["fold"].nunique() != 1).any():
        raise IntegrityError("A patient is assigned to multiple frozen validation folds")
