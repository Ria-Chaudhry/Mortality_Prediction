"""Deterministic patient-grouped five-fold assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupKFold

from ..errors import IntegrityError
from ..hashing import hash_frame


@dataclass
class FoldResult:
    assignments: pd.DataFrame
    public_summary: pd.DataFrame
    fold_hash: str


def create_patient_folds(cohort: pd.DataFrame, config: dict[str, Any]) -> FoldResult:
    count = int(config["folds"]["count"])
    seed = int(config["folds"]["seed"])
    working = cohort[
        ["cohort_visit_number", "patient_id", "outcome"]
    ].copy()
    working["_patient_order"] = working["patient_id"].map(
        lambda value: _stable_key(str(value), seed)
    )
    working = working.sort_values(
        ["_patient_order", "patient_id", "cohort_visit_number"], kind="stable"
    ).reset_index(drop=True)
    splitter = GroupKFold(n_splits=count, shuffle=True, random_state=seed)
    working["fold"] = -1
    for fold, (_, validation) in enumerate(
        splitter.split(working, working["outcome"], groups=working["patient_id"])
    ):
        working.loc[validation, "fold"] = fold
    if (working["fold"] < 0).any():
        raise IntegrityError("A visit has no fold assignment")
    patient_fold_counts = working.groupby("patient_id")["fold"].nunique()
    if (patient_fold_counts != 1).any():
        raise IntegrityError("A patient is assigned to multiple validation folds")
    if config["folds"].get("require_both_classes", True):
        for fold in range(count):
            validation_classes = working.loc[working["fold"] == fold, "outcome"].nunique()
            training_classes = working.loc[working["fold"] != fold, "outcome"].nunique()
            if validation_classes != 2 or training_classes != 2:
                raise IntegrityError(
                    f"Fold {fold} does not contain both outcome classes in train and validation"
                )
    assignments = working[
        ["cohort_visit_number", "patient_id", "fold"]
    ].sort_values("cohort_visit_number", kind="stable")
    summary = (
        working.groupby("fold", sort=True)
        .agg(visits=("cohort_visit_number", "size"), patients=("patient_id", "nunique"), events=("outcome", "sum"))
        .reset_index()
    )
    fold_hash = hash_frame(assignments, ["cohort_visit_number", "patient_id", "fold"])
    return FoldResult(assignments=assignments, public_summary=summary, fold_hash=fold_hash)


def _stable_key(value: str, seed: int) -> str:
    import hashlib

    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()
