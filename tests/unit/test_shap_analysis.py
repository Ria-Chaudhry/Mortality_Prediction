from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from clinical_domain_mortality.modeling import fold_shap_aggregate


def test_shap_uses_training_background_and_held_out_evaluation(chorus_config):
    config = deepcopy(chorus_config)
    config["models"]["shap"]["background_rows"] = 4
    config["models"]["shap"]["evaluation_rows"] = 3
    x_train = pd.DataFrame(
        {
            "age": [40, 50, 60, 70, 80, 55],
            "sex": ["F", "M", "F", "M", "F", "M"],
            "measurement__sodium__mean": [135, 140, 132, 145, 138, 141],
            "measurement__sodium__count": [1, 2, 1, 3, 2, 1],
        }
    )
    y_train = pd.Series([0, 0, 1, 1, 1, 0], dtype=int)
    x_validation = pd.DataFrame(
        {
            "age": [45, 65, 75],
            "sex": ["F", "M", "F"],
            "measurement__sodium__mean": [136, 142, 139],
            "measurement__sodium__count": [1, 2, 1],
        }
    )
    result = fold_shap_aggregate(
        x_train,
        y_train,
        x_validation,
        dataset="synthetic",
        fold=0,
        matrix="baseline_measurements",
        model="logistic_regression",
        config=config,
    )
    repeated = fold_shap_aggregate(
        x_train,
        y_train,
        x_validation,
        dataset="synthetic",
        fold=0,
        matrix="baseline_measurements",
        model="logistic_regression",
        config=config,
    )
    pd.testing.assert_frame_equal(result, repeated, check_exact=True)
    assert set(result["feature"]) == set(x_train.columns)
    assert result["mean_absolute_shap"].ge(0).all()
    assert result["rank"].tolist() == list(range(1, len(result) + 1))
    assert set(result["evaluation_partition"]) == {"outer_validation_fold"}
    assert set(result["background_partition"]) == {"outer_training_fold"}
    assert np.isfinite(result["mean_absolute_shap"]).all()
    sodium = result.loc[
        result["feature"].eq("measurement__sodium__mean")
    ].iloc[0]
    assert sodium["clinical_domain"] == "measurements"
    assert sodium["source_concept"] == "sodium"
    assert sodium["summary_type"] == "mean"
