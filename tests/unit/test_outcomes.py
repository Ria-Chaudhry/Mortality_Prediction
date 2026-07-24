import pandas as pd

from clinical_domains.core.outcomes import build_mortality_labels


def test_build_mortality_labels_defaults_missing_to_zero():
    encounters = pd.DataFrame({"encounter_id": ["e1", "e2"], "patient_id": ["p1", "p2"]})
    mortality = pd.DataFrame({"encounter_id": ["e1"], "died": [1]})

    labels = build_mortality_labels(encounters, mortality)

    assert labels.set_index("encounter_id").loc["e1", "died"] == 1
    assert labels.set_index("encounter_id").loc["e2", "died"] == 0
