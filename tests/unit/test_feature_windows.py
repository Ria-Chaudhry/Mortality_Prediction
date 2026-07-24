import pandas as pd

from clinical_domains.features.aggregation import aggregate_events


def test_aggregate_events_names_domain_feature_stat_columns():
    events = pd.DataFrame(
        {
            "encounter_id": ["e1", "e1"],
            "event_time": ["2024-01-01 01:00:00", "2024-01-01 02:00:00"],
            "domain": ["physiological", "physiological"],
            "feature_name": ["heart_rate", "heart_rate"],
            "value": [80, 100],
        }
    )

    result = aggregate_events(events)

    assert result.loc[0, "physiological__heart_rate__mean"] == 90
    assert result.loc[0, "physiological__heart_rate__last"] == 100
