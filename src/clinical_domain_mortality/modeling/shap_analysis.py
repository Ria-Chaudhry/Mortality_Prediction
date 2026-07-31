"""Restricted fold-level and safe aggregate SHAP analysis."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from ..errors import IntegrityError
from ..hashing import hash_object
from .runner import fit_predict_fold


def fold_shap_aggregate(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    *,
    dataset: str,
    fold: int,
    matrix: str,
    model: str,
    config: dict[str, Any],
    fitted_pipeline: Pipeline | None = None,
    model_fit_source: str = "fit_inside_shap_stage",
) -> pd.DataFrame:
    """Explain held-out rows against a training-only background.

    The permutation explainer and training/validation sampling reproduce the
    recoverable structure of the completed MIMIC scripts. Only mean absolute
    fold aggregates leave this function; encounter-level SHAP values are never
    returned or written.
    """
    settings = config["models"]["shap"]
    if settings.get("explainer") != "permutation":
        raise IntegrityError("Only the recovered permutation SHAP procedure is supported")
    if list(x_train.columns) != list(x_validation.columns):
        raise IntegrityError(
            "SHAP training and held-out input feature order differs"
        )
    pipeline = (
        fitted_pipeline
        if fitted_pipeline is not None
        else fit_predict_fold(
            x_train, y_train, x_validation, model, config
        ).pipeline
    )
    fitted_names = getattr(pipeline, "feature_names_in_", None)
    if fitted_names is None or list(map(str, fitted_names)) != list(
        x_train.columns.astype(str)
    ):
        raise IntegrityError(
            "Fitted model feature order differs from the SHAP matrix"
        )
    transformed_train = _transform_without_estimator(pipeline, x_train)
    transformed_validation = _transform_without_estimator(pipeline, x_validation)
    transformed_names, raw_names = _retained_feature_names(pipeline, x_train)
    if transformed_train.shape[1] != len(transformed_names):
        raise IntegrityError("SHAP transformed feature names do not match matrix width")

    seed = int(settings["seed"]) + int(fold)
    background_positions = _deterministic_sample_positions(
        len(transformed_train),
        int(settings["background_rows"]),
        seed,
    )
    evaluation_positions = _deterministic_sample_positions(
        len(transformed_validation),
        int(settings["evaluation_rows"]),
        seed + int(settings.get("evaluation_seed_offset", 100)),
    )
    background = transformed_train[background_positions]
    explained = transformed_validation[evaluation_positions]
    estimator = pipeline.named_steps["estimator"]
    classes = np.asarray(pipeline.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise IntegrityError("SHAP cannot identify the positive class")

    def predict_positive(values: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            return np.asarray(
                estimator.predict_proba(values)[:, positive[0]], dtype=float
            )

    explainer = shap.PermutationExplainer(
        predict_positive,
        background,
        feature_names=transformed_names,
        seed=seed,
    )
    max_evals = max(
        2 * transformed_train.shape[1] + 1,
        int(settings.get("minimum_max_evals", 101)),
    )
    explanation = explainer(
        explained,
        max_evals=max_evals,
        batch_size=int(settings.get("batch_size", 20)),
        silent=True,
    )
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(raw_names):
        raise IntegrityError("SHAP explanation has an unexpected shape")
    transformed_mean = np.mean(np.abs(values), axis=0)
    aggregate = (
        pd.DataFrame(
            {
                "feature": raw_names,
                "mean_absolute_shap": transformed_mean,
            }
        )
        .groupby("feature", sort=False, as_index=False)["mean_absolute_shap"]
        .sum()
    )
    provenance = aggregate["feature"].map(_raw_feature_provenance)
    aggregate["clinical_domain"] = provenance.map(lambda item: item[0])
    aggregate["source_concept"] = provenance.map(lambda item: item[1])
    aggregate["summary_type"] = provenance.map(lambda item: item[2])
    aggregate.insert(0, "model", model)
    aggregate.insert(0, "feature_matrix", matrix)
    aggregate.insert(0, "outer_fold", fold)
    aggregate.insert(0, "dataset", dataset)
    aggregate = aggregate.sort_values(
        ["mean_absolute_shap", "feature"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    aggregate["rank"] = np.arange(1, len(aggregate) + 1, dtype=np.int64)
    aggregate["explainer"] = "permutation"
    aggregate["background_partition"] = "outer_training_fold"
    aggregate["evaluation_partition"] = "outer_validation_fold"
    aggregate["background_rows"] = len(background)
    aggregate["evaluation_rows"] = len(explained)
    aggregate["random_seed"] = seed
    aggregate["model_fit_source"] = model_fit_source
    aggregate["input_feature_count"] = len(x_train.columns)
    aggregate["input_feature_order_hash"] = hash_object(
        x_train.columns.astype(str).tolist()
    )
    aggregate["selected_input_features"] = "|".join(
        x_train.columns.astype(str).tolist()
    )
    aggregate["encoded_feature_count"] = len(transformed_names)
    aggregate["training_partition_index_hash"] = hash_object(
        x_train.index.tolist()
    )
    aggregate["validation_partition_index_hash"] = hash_object(
        x_validation.index.tolist()
    )
    aggregate["background_position_hash"] = hash_object(
        background_positions.tolist()
    )
    aggregate["evaluation_position_hash"] = hash_object(
        evaluation_positions.tolist()
    )
    aggregate["fold_aggregation_policy"] = str(
        settings.get(
            "cross_fold_aggregation",
            "mean_over_folds_where_feature_selected_v1",
        )
    )
    return aggregate


def _transform_without_estimator(
    pipeline: Pipeline, frame: pd.DataFrame
) -> np.ndarray:
    transformed = pipeline.named_steps["preprocessing"].transform(frame)
    transformed = pipeline.named_steps["variance"].transform(transformed)
    if "scaler" in pipeline.named_steps:
        transformed = pipeline.named_steps["scaler"].transform(transformed)
    return np.asarray(transformed, dtype=float)


def _retained_feature_names(
    pipeline: Pipeline, frame: pd.DataFrame
) -> tuple[list[str], list[str]]:
    preprocessing = pipeline.named_steps["preprocessing"]
    transformed = preprocessing.get_feature_names_out().astype(str).tolist()
    raw: list[str] = []
    for transformed_name in transformed:
        stripped = transformed_name.split("__", 1)[-1]
        match = next(
            (
                name
                for name in frame.columns
                if stripped == name or stripped.startswith(f"{name}_")
            ),
            None,
        )
        if match is None:
            raise IntegrityError(
                f"Cannot map transformed SHAP feature to an input column: {transformed_name}"
            )
        raw.append(match)
    support = pipeline.named_steps["variance"].get_support()
    return (
        [name for name, keep in zip(transformed, support, strict=True) if keep],
        [name for name, keep in zip(raw, support, strict=True) if keep],
    )


def _deterministic_sample_positions(
    row_count: int, limit: int, seed: int
) -> np.ndarray:
    if row_count <= limit:
        return np.arange(row_count, dtype=np.int64)
    positions = np.random.default_rng(seed).choice(
        row_count, size=limit, replace=False
    )
    return np.sort(positions).astype(np.int64)


def _raw_feature_provenance(name: str) -> tuple[str, str, str]:
    for prefix, domain in (
        ("measurement__", "measurements"),
        ("medication__", "medications"),
        ("procedure__", "procedures"),
    ):
        if name.startswith(prefix):
            concept, separator, summary = name.removeprefix(prefix).rpartition("__")
            if separator:
                return domain, concept, summary
    for aggregate, domain in (
        ("drug", "medications"),
        ("procedure", "procedures"),
    ):
        if aggregate in name:
            return domain, "__domain_aggregate__", name
    return "baseline", "__baseline__", name
