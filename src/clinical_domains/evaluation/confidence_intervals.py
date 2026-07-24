from __future__ import annotations

import numpy as np


def percentile_interval(values, alpha: float = 0.05) -> tuple[float, float]:
    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)
    return tuple(float(x) for x in np.percentile(values, [lower, upper]))
