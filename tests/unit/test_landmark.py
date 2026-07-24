import pandas as pd

from clinical_domains.core.landmark import restrict_events_to_landmark


def test_restrict_events_to_landmark_keeps_only_early_events():
    encounters = pd.DataFrame(
        {
            "encounter_id": ["e1"],
            "admit_time": ["2024-01-01 00:00:00"],
        }
    )
    events = pd.DataFrame(
        {
            "encounter_id": ["e1", "e1", "e1"],
            "event_time": [
                "2023-12-31 23:00:00",
                "2024-01-01 02:00:00",
                "2024-01-02 12:00:00",
            ],
            "domain": ["physiological"] * 3,
            "feature_name": ["heart_rate"] * 3,
            "value": [80, 90, 100],
        }
    )

    result = restrict_events_to_landmark(events, encounters, hours=24)

    assert list(result["value"]) == [90]
    assert result["hours_from_admit"].iloc[0] == 2
