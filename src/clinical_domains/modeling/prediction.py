from __future__ import annotations

import numpy as np


def predict_probability(model, features) -> np.ndarray:
    return model.predict_proba(features)[:, 1]
