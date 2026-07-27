from __future__ import annotations

import pandas as pd
import pytest

from clinical_domain_mortality.cohort import create_patient_folds


def test_cohort_landmark_outcome_and_early_death(cohort_result):
    cohort = cohort_result.cohort
    assert len(cohort) == 70
    assert cohort["patient_id"].nunique() == 70
    assert not set(f"P{i:03d}" for i in range(5)) & set(cohort["patient_id"])
    assert (cohort["landmark_datetime"] - cohort["start_datetime"]).eq(
        pd.Timedelta(hours=24)
    ).all()
    assert int(cohort["outcome"].sum()) == 15
    events = cohort.loc[cohort["outcome"] == 1]
    assert (events["death_datetime"] > events["landmark_datetime"]).all()
    assert (events["death_datetime"] <= events["outcome_horizon_datetime"]).all()


def test_short_visits_retained_and_window_ends_at_discharge(cohort_result):
    short = cohort_result.cohort.loc[cohort_result.cohort["short_visit"]]
    assert not short.empty
    assert short["predictor_end_datetime"].equals(short["end_datetime"])


def test_charlson_uses_prior_encounters_only(cohort_result):
    row = cohort_result.cohort.set_index("patient_id").loc["P006"]
    assert row["prior_charlson_score"] == 2
    assert row["prior_charlson_score"] != 8


def test_row_order_and_hash_are_stable(cohort_result):
    cohort = cohort_result.cohort
    assert cohort["cohort_visit_number"].tolist() == list(range(1, 71))
    assert len(cohort_result.cohort_hash) == 64
    assert len(cohort_result.row_order_hash) == 64


def test_patient_folds_are_isolated_and_complete(fold_result, cohort_result):
    assignments = fold_result.assignments
    assert assignments["fold"].nunique() == 5
    assert assignments.groupby("patient_id")["fold"].nunique().max() == 1
    assert set(assignments["cohort_visit_number"]) == set(
        cohort_result.cohort["cohort_visit_number"]
    )


def test_fold_assignment_is_deterministic(cohort_result, chorus_config, fold_result):
    repeated = create_patient_folds(cohort_result.cohort.sample(frac=1, random_state=9), chorus_config)
    left = fold_result.assignments.sort_values("cohort_visit_number").reset_index(drop=True)
    right = repeated.assignments.sort_values("cohort_visit_number").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert repeated.fold_hash == fold_result.fold_hash


def test_fold_integrity_hard_fails_for_too_few_groups(chorus_config):
    cohort = pd.DataFrame(
        {
            "cohort_visit_number": [1, 2, 3, 4],
            "patient_id": ["a", "b", "c", "d"],
            "outcome": [0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError):
        create_patient_folds(cohort, chorus_config)
