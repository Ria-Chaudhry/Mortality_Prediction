from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from clinical_domain_mortality.cohort import build_cohort, create_patient_folds
from clinical_domain_mortality.features import prepare_domain_events


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


def test_predictor_window_changes_events_without_changing_landmark(
    chorus_data, mutable_config
):
    mutable_config["cohort"]["predictor_window_hours"] = 6
    cohort = build_cohort(chorus_data, mutable_config)
    prepared = prepare_domain_events(chorus_data, cohort.cohort, mutable_config)
    assert (
        cohort.cohort["landmark_datetime"] - cohort.cohort["start_datetime"]
    ).eq(pd.Timedelta(hours=24)).all()
    assert (
        cohort.cohort["configured_predictor_end_datetime"]
        - cohort.cohort["start_datetime"]
    ).eq(pd.Timedelta(hours=6)).all()
    for frame in prepared.events.values():
        assert (frame["hours_from_start"] < 6).all()


@pytest.mark.parametrize(
    ("precision", "offset", "expected_present", "expected_outcome"),
    [
        ("datetime", pd.Timedelta(hours=26), True, 1),
        ("datetime", pd.Timedelta(hours=24), False, None),
        ("datetime", pd.Timedelta(days=31), True, 0),
        ("date", pd.Timedelta(days=1), False, None),
        ("date", pd.Timedelta(days=5), True, 1),
        (None, None, True, 0),
    ],
)
def test_death_precision_boundaries_are_used_consistently(
    chorus_data,
    chorus_config,
    precision,
    offset,
    expected_present,
    expected_outcome,
):
    data = deepcopy(chorus_data)
    patient = "P073"
    visit = data.tables["encounters"].loc[
        data.tables["encounters"]["patient_id"].eq(patient)
    ].iloc[0]
    deaths = data.tables["deaths"].loc[
        ~data.tables["deaths"]["patient_id"].eq(patient)
    ].copy()
    if precision is not None:
        value = visit["start_datetime"] + offset
        deaths = pd.concat(
            [
                deaths,
                pd.DataFrame(
                    [
                        {
                            "patient_id": patient,
                            "death_datetime": value if precision == "datetime" else pd.NaT,
                            "death_date": value.normalize(),
                            "death_time_precision": precision,
                            "death_source": (
                                "admissions.deathtime"
                                if precision == "datetime"
                                else "patients.dod"
                            ),
                            "death_source_conflict": False,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    data.tables["deaths"] = deaths
    result = build_cohort(data, chorus_config).cohort
    observed = result.loc[result["patient_id"].eq(patient)]
    assert (not observed.empty) is expected_present
    if expected_present:
        assert int(observed.iloc[0]["outcome"]) == expected_outcome
