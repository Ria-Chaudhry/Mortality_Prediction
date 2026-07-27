"""Patient-clustered percentile bootstrap shared across analyses."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def bootstrap_indices(
    patient_ids: pd.Series,
    repetitions: int,
    seed: int,
) -> list[np.ndarray]:
    """Sample patients with replacement and include every visit for each draw."""
    patients = np.asarray(sorted(patient_ids.astype(str).unique()))
    rows_by_patient = {
        patient: np.flatnonzero(patient_ids.astype(str).to_numpy() == patient)
        for patient in patients
    }
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(repetitions):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        output.append(np.concatenate([rows_by_patient[patient] for patient in sampled]))
    return output


def percentile_interval(
    values: Iterable[float], confidence_level: float
) -> tuple[float, float, int, int]:
    values_array = np.asarray(list(values), dtype=float)
    valid = values_array[np.isfinite(values_array)]
    invalid = int(len(values_array) - len(valid))
    if len(valid) == 0:
        return float("nan"), float("nan"), invalid, 0
    alpha = 1 - confidence_level
    lower, upper = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper), invalid, len(valid)
