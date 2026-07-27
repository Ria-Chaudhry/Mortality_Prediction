"""Shared synthetic scientific objects."""

from __future__ import annotations

from copy import deepcopy

import pytest

from clinical_domain_mortality.adapters import CHoRUSAdapter, MIMICIVAdapter
from clinical_domain_mortality.cohort import build_cohort, create_patient_folds
from clinical_domain_mortality.config import PROJECT_ROOT, load_config
from clinical_domain_mortality.features import prepare_domain_events


@pytest.fixture(scope="session")
def chorus_config():
    return load_config(PROJECT_ROOT / "configs" / "chorus.example.yaml")


@pytest.fixture(scope="session")
def mimic_config():
    return load_config(PROJECT_ROOT / "configs" / "mimic.example.yaml")


@pytest.fixture(scope="session")
def chorus_data(chorus_config):
    return CHoRUSAdapter(chorus_config).load()


@pytest.fixture(scope="session")
def mimic_data(mimic_config):
    return MIMICIVAdapter(mimic_config).load()


@pytest.fixture(scope="session")
def cohort_result(chorus_data, chorus_config):
    return build_cohort(chorus_data, chorus_config)


@pytest.fixture(scope="session")
def fold_result(cohort_result, chorus_config):
    return create_patient_folds(cohort_result.cohort, chorus_config)


@pytest.fixture(scope="session")
def prepared_events(chorus_data, cohort_result, chorus_config):
    return prepare_domain_events(chorus_data, cohort_result.cohort, chorus_config)


@pytest.fixture
def mutable_config(chorus_config):
    return deepcopy(chorus_config)
