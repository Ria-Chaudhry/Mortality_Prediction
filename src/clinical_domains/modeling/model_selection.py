from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from clinical_domains.modeling.algorithms import get_classifier
from clinical_domains.modeling.cross_validation import grouped_cv_splits
from clinical_domains.preprocessing.leakage_checks import assert_no_outcome_columns
from clinical_domains.preprocessing.pipelines import build_preprocessor


def cross_validated_predictions(
    matrix: pd.DataFrame,
    feature_columns: list[str],
    outcome_col: str = "outcome",
    group_col: str = "patient_id",
    model_name: str = "logistic_regression",
    n_splits: int = 5,
    seed: int = 20260724,
) -> pd.DataFrame:
    """Fit preprocessing and model steps inside each grouped training fold."""
    assert_no_outcome_columns(feature_columns)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        grouped_cv_splits(matrix, group_col=group_col, target_col=outcome_col, n_splits=n_splits, seed=seed)
    ):
        train = matrix.iloc[train_idx]
        test = matrix.iloc[test_idx]
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(train[feature_columns])),
                ("model", get_classifier(model_name, seed=seed)),
            ]
        )
        pipeline.fit(train[feature_columns], train[outcome_col])
        scores = pipeline.predict_proba(test[feature_columns])[:, 1]
        fold_rows = test[["encounter_id", group_col, outcome_col]].copy()
        fold_rows["y_score"] = scores
        fold_rows["fold"] = fold
        fold_rows["model_name"] = model_name
        rows.append(fold_rows)
    return pd.concat(rows, ignore_index=True)
