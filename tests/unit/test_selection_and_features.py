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
    ("domain", "constructed"),
    [("measurements", 300), ("medications", 104), ("procedures", 103)],
)
def test_expected_domain_feature_counts(
    domain, constructed, prepared_events, fold_result, cohort_result, chorus_config
):
    training = _training_sets(fold_result, 0)
    selection = select_concepts(
        prepared_events.events[domain],
        training,
        domain,
        0,
        chorus_config,
    )
    feature = build_fold_domain_features(
        cohort_result.cohort,
        selection,
        prepared_events.events[domain],
        training,
        chorus_config,
    )
    assert feature.full_feature_count == constructed
    assert len(feature.feature_names) == 21
    assert len(feature.selection_audit) == constructed
    selected = feature.selection_audit.loc[feature.selection_audit["selected"]]
    assert len(selected) == 21
    assert selected["candidate_feature_name"].tolist() == feature.feature_names
    assert selected["source_concept"].nunique() < 21
    assert set(feature.selection_audit["selection_rule_identifier"]) == {
        "training_support_prevalence_v1"
    }
    assert len(feature.frame) == len(cohort_result.cohort)
    assert feature.frame["cohort_visit_number"].tolist() == cohort_result.cohort[
        "cohort_visit_number"
    ].tolist()


def test_zero_event_visits_are_preserved(
    prepared_events, fold_result, cohort_result, chorus_config
):
    training = _training_sets(fold_result, 0)
    selection = select_concepts(
        prepared_events.events["medications"],
        training,
        "medications",
        0,
        chorus_config,
    )
    feature = build_fold_domain_features(
        cohort_result.cohort,
        selection,
        prepared_events.events["medications"],
        training,
        chorus_config,
    ).frame.set_index("cohort_visit_number")
    no_event_numbers = cohort_result.cohort.loc[
        cohort_result.cohort["patient_id"].isin(["P073", "P074"]),
        "cohort_visit_number",
    ]
    values = feature.loc[no_event_numbers]
    assert values.fillna(0).eq(0).all().all()


def test_derived_float_features_are_canonicalized_before_modeling(
    prepared_events, fold_result, cohort_result, chorus_config
):
    training = _training_sets(fold_result, 0)
    events = prepared_events.events["measurements"].copy()
    selected_concept = events.loc[
        events["cohort_visit_number"].isin(training), "concept_key"
    ].astype(str).iloc[0]
    mask = (
        events["cohort_visit_number"].isin(training)
        & events["concept_key"].astype(str).eq(selected_concept)
    )
    events.loc[mask, "value_numeric"] = 1.123456789123
    selection = select_concepts(
        events, training, "measurements", 0, chorus_config
    )
    feature = build_fold_domain_features(
        cohort_result.cohort,
        selection,
        events,
        training,
        chorus_config,
    )
    decimal_places = chorus_config["features"]["numeric_canonicalization"][
        "decimal_places"
    ]
    float_columns = [
        column
        for column in feature.frame
        if pd.api.types.is_float_dtype(feature.frame[column])
    ]
    assert float_columns
    for column in float_columns:
        observed = feature.frame[column].dropna()
        assert observed.eq(observed.round(decimal_places)).all()


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
    domain = DomainFeatures(
        domain="measurements",
        fold=0,
        frame=lost,
        feature_names=["feature"],
        full_feature_count=1,
        selection_audit=pd.DataFrame(),
        feature_schema_hash="schema",
        feature_value_hash="values",
    )
    with pytest.raises(IntegrityError):
        assemble_matrix(
            baseline,
            {"measurements": domain},
            "baseline_measurements",
            chorus_config,
        )


@pytest.mark.parametrize("domain", ["measurements", "medications", "procedures"])
def test_validation_changes_cannot_change_top50_or_top21(
    domain, prepared_events, fold_result, cohort_result, chorus_config
):
    fold = 0
    training = _training_sets(fold_result, fold)
    validation = set(
        fold_result.assignments.loc[
            fold_result.assignments["fold"] == fold, "cohort_visit_number"
        ]
    )
    original_events = prepared_events.events[domain]
    original_selection = select_concepts(
        original_events, training, domain, fold, chorus_config
    )
    original_features = build_fold_domain_features(
        cohort_result.cohort,
        original_selection,
        original_events,
        training,
        chorus_config,
    )
    changed = original_events.copy()
    validation_mask = changed["cohort_visit_number"].isin(validation)
    changed.loc[validation_mask, "concept_key"] = "validation_only_replacement"
    changed.loc[validation_mask, "concept_name"] = "validation only"
    changed.loc[validation_mask, "value"] = 1e12
    changed_selection = select_concepts(
        changed, training, domain, fold, chorus_config
    )
    changed_features = build_fold_domain_features(
        cohort_result.cohort,
        changed_selection,
        changed,
        training,
        chorus_config,
    )
    outcome_changed_cohort = cohort_result.cohort.copy()
    outcome_changed_cohort.loc[
        outcome_changed_cohort["cohort_visit_number"].isin(validation), "outcome"
    ] = 1 - outcome_changed_cohort.loc[
        outcome_changed_cohort["cohort_visit_number"].isin(validation), "outcome"
    ]
    outcome_changed_features = build_fold_domain_features(
        outcome_changed_cohort,
        original_selection,
        original_events,
        training,
        chorus_config,
    )
    assert original_selection.selected["concept_key"].tolist() == changed_selection.selected[
        "concept_key"
    ].tolist()
    assert original_features.feature_names == changed_features.feature_names
    assert (
        original_features.selection_audit["training_support_count"].tolist()
        == changed_features.selection_audit["training_support_count"].tolist()
    )
    assert original_features.feature_names == outcome_changed_features.feature_names


def test_domain_feature_definitions_are_reused_across_matrices(
    prepared_events, fold_result, cohort_result, chorus_config
):
    training = _training_sets(fold_result, 0)
    domains = {}
    for domain in ("measurements", "medications", "procedures"):
        selection = select_concepts(
            prepared_events.events[domain], training, domain, 0, chorus_config
        )
        domains[domain] = build_fold_domain_features(
            cohort_result.cohort,
            selection,
            prepared_events.events[domain],
            training,
            chorus_config,
        )
    for domain in domains:
        expected = domains[domain].feature_names
        for matrix_name, components in chorus_config["matrices"].items():
            if domain in components:
                matrix = assemble_matrix(
                    cohort_result.baseline, domains, matrix_name, chorus_config
                )
                actual = [column for column in matrix.columns if column in expected]
                assert actual == expected
                assert len(actual) == 21


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
