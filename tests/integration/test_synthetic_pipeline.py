from pathlib import Path

from clinical_domains.adapters.generic_ehr import GenericEHRAdapter
from clinical_domains.core.cohort import build_cohort
from clinical_domains.core.landmark import restrict_events_to_landmark
from clinical_domains.core.outcomes import build_mortality_labels
from clinical_domains.features.aggregation import aggregate_events
from clinical_domains.features.matrices import build_feature_matrix


def test_synthetic_pipeline_builds_standard_matrix():
    config = Path("examples/synthetic/config.yaml")
    data = GenericEHRAdapter.from_config(config).extract_all()

    cohort = build_cohort(data["encounters"])
    events = restrict_events_to_landmark(data["events"], cohort, hours=24)
    event_features = aggregate_events(events)
    matrix = build_feature_matrix(cohort, data["baseline"], event_features)
    labels = build_mortality_labels(cohort, data["mortality"])

    assert len(matrix) == len(labels) == 6
    assert "baseline__age" in matrix.columns
    assert "physiological__heart_rate__mean" in matrix.columns
