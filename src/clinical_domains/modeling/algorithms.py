from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_classifier(name: str, seed: int = 20260724, **kwargs):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=kwargs.pop("max_iter", 1000), **kwargs)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **kwargs)
    raise ValueError(f"Unknown classifier: {name}")
