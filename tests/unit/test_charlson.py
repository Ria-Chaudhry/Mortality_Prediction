from __future__ import annotations

import pandas as pd
import pytest

from clinical_domain_mortality.cohort.builder import _prior_features
from clinical_domain_mortality.cohort.charlson import (
    ALGORITHM_VERSION,
    classify_icd,
    normalize_icd_code,
    score_diagnosis_frame,
)
from clinical_domain_mortality.errors import IntegrityError


@pytest.mark.parametrize(
    ("code", "version", "category"),
    [
        ("428.0", 9, "congestive_heart_failure"),
        ("250.40", 9, "diabetes_with_complication"),
        ("I50.9", 10, "congestive_heart_failure"),
        ("C78.7", 10, "metastatic_solid_tumor"),
    ],
)
def test_charlson_icd9_and_icd10_examples(code, version, category):
    assert category in classify_icd(code, version)
    assert ALGORITHM_VERSION == "quan-2005-charlson-icd9cm-icd10cm-v1"


def test_charlson_normalizes_punctuation_and_case_without_truncation():
    assert normalize_icd_code(" e11.22 ") == "E1122"
    assert "diabetes_with_complication" in classify_icd(" e11.22 ", 10)


def test_charlson_hierarchies_and_duplicate_diagnoses():
    frame = pd.DataFrame(
        {
            "code": ["E11.9", "E11.22", "K73.9", "K72.1", "C50.9", "C78.7", "C78.7"],
            "icd_version": [10] * 7,
        }
    )
    result = score_diagnosis_frame(frame)
    assert "diabetes_without_complication" not in result.categories
    assert "mild_liver_disease" not in result.categories
    assert "nonmetastatic_malignancy" not in result.categories
    assert result.score == 2 + 3 + 6


def test_charlson_requires_valid_icd_version():
    with pytest.raises(IntegrityError, match="ICD version"):
        classify_icd("I50", None)


def test_prior_only_365_day_boundary_current_exclusion_and_multi_admission(
    chorus_config,
):
    index_start = pd.Timestamp("2020-01-01 08:00:00")
    cohort = pd.DataFrame(
        {
            "cohort_visit_number": [1],
            "patient_id": ["p1"],
            "start_datetime": [index_start],
        }
    )
    encounters = pd.DataFrame(
        {
            "visit_id": ["boundary", "older", "second", "index"],
            "patient_id": ["p1"] * 4,
            "start_datetime": [
                index_start - pd.Timedelta(days=365),
                index_start - pd.Timedelta(days=365, seconds=1),
                index_start - pd.Timedelta(days=10),
                index_start,
            ],
            "visit_type": ["inpatient"] * 4,
        }
    )
    diagnoses = pd.DataFrame(
        {
            "visit_id": ["boundary", "boundary", "older", "second", "index"],
            "code": ["N18.6", "N18.6", "C78.7", "I50.9", "C78.7"],
            "icd_version": [10] * 5,
        }
    )
    result = _prior_features(
        cohort,
        encounters,
        diagnoses,
        chorus_config["cohort"],
        chorus_config,
    ).iloc[0]
    assert result["prior_visit_count"] == 2
    assert result["prior_charlson_score"] == 3
