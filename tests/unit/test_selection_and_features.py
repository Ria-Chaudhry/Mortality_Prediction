from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from clinical_domain_mortality.errors import IntegrityError, UnitError
from clinical_domain_mortality.features import (
    DomainFeatures,
    assemble_matrix,
    build_fold_domain_features,
    select_concepts,
)


def _training_sets(fold_result, fold):
    assignments = fold_result.assignments
    return set(
        assignments.loc[assignments["fold"] != fold, "cohort_visit_number"].astype(int)
    )


def test_concepts_are_selected_inside_each_training_fold(
    prepared_events, fold_result, chorus_config
):
    selections = []
    for fold in range(5):
        selection = select_concepts(
            prepared_events.events["measurements"],
            _training_sets(fold_result, fold),
            "measurements",
            fold,
            chorus_config,
        )
        selections.append(tuple(selection.selected["concept_key"]))
    assert len(set(selections)) > 1
    assert all(len(item) == 50 for item in selections)


def test_validation_only_prevalence_cannot_change_selection(chorus_config):
    rows = []
    for visit in range(1, 5):
        for concept in range(50):
            rows.append(_event(visit, f"c{concept:02d}"))
    for _ in range(500):
        rows.append(_event(5, "validation_only"))
    selection = select_concepts(
        pd.DataFrame(rows), {1, 2, 3, 4}, "measurements", 0, chorus_config
    )
    assert "validation_only" not in set(selection.selected["concept_key"])


def test_concept_ties_break_by_normalized_key(chorus_config):
    rows = []
    for concept in reversed(range(50)):
        rows.append(_event(1, f"C{concept:02d}"))
    selection = select_concepts(
        pd.DataFrame(rows), {1}, "measurements", 0, chorus_config
    )
    assert selection.selected.iloc[0]["concept_key"] == "C00"


def test_incompatible_units_hard_fail(chorus_config):
    config = deepcopy(chorus_config)
    config["features"]["measurements"]["allowed_units"]["default"] = ["u", "other"]
    rows = []
    for concept in range(50):
        rows.append(_event(1, f"c{concept:02d}", unit="u"))
    rows.append(_event(2, "c00", unit="other"))
    with pytest.raises(UnitError):
        select_concepts(
            pd.DataFrame(rows), {1, 2}, "measurements", 0, config
        )


@pytest.mark.parametrize(
    ("domain", "expected"),
    [("measurements", 300), ("medications", 104), ("procedures", 103)],
)
def test_expected_domain_feature_counts(
    domain, expected, prepared_events, fold_result, cohort_result, chorus_config
):
    selection = select_concepts(
        prepared_events.events[domain],
        _training_sets(fold_result, 0),
        domain,
        0,
        chorus_config,
    )
    feature = build_fold_domain_features(
        cohort_result.cohort,
        selection,
        prepared_events.events[domain],
        chorus_config,
    )
    assert len(feature.feature_names) == expected
    assert len(feature.frame) == len(cohort_result.cohort)
    assert feature.frame["cohort_visit_number"].tolist() == cohort_result.cohort[
        "cohort_visit_number"
    ].tolist()


def test_zero_event_visits_are_preserved(
    prepared_events, fold_result, cohort_result, chorus_config
):
    selection = select_concepts(
        prepared_events.events["medications"],
        _training_sets(fold_result, 0),
        "medications",
        0,
        chorus_config,
    )
    feature = build_fold_domain_features(
        cohort_result.cohort,
        selection,
        prepared_events.events["medications"],
        chorus_config,
    ).frame.set_index("cohort_visit_number")
    no_event_numbers = cohort_result.cohort.loc[
        cohort_result.cohort["patient_id"].isin(["P073", "P074"]),
        "cohort_visit_number",
    ]
    assert (feature.loc[no_event_numbers, "any_drug_24h"] == 0).all()
    assert feature.loc[no_event_numbers, "time_to_first_drug_in_hours"].isna().all()


def test_too_few_concepts_hard_fails(chorus_config):
    with pytest.raises(IntegrityError):
        select_concepts(
            pd.DataFrame([_event(1, "only")]),
            {1},
            "measurements",
            0,
            chorus_config,
        )


def test_row_loss_in_domain_matrix_hard_fails(cohort_result, chorus_config):
    baseline = cohort_result.baseline
    lost = pd.DataFrame(
        {
            "cohort_visit_number": baseline["cohort_visit_number"].iloc[:-1],
            "feature": 1,
        }
    )
    domain = DomainFeatures("measurements", 0, lost, ["feature"], "hash")
    with pytest.raises(IntegrityError):
        assemble_matrix(
            baseline,
            {"measurements": domain},
            "baseline_measurements",
            chorus_config,
        )


def _event(visit, concept, unit="u"):
    return {
        "event_id": f"{visit}-{concept}-{unit}",
        "cohort_visit_number": visit,
        "concept_key": concept,
        "concept_name": concept,
        "value": 1.0,
        "unit": unit,
        "source_table": "test",
        "semantics": "result",
    }
