import pandas as pd

from clinical_domains.modeling.cross_validation import grouped_cv_splits


def test_grouped_cv_never_splits_patient_across_train_and_test():
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "outcome": [0, 0, 1, 1, 0, 1],
        }
    )

    for train_idx, test_idx in grouped_cv_splits(frame, n_splits=3):
        train_groups = set(frame.iloc[train_idx]["patient_id"])
        test_groups = set(frame.iloc[test_idx]["patient_id"])
        assert train_groups.isdisjoint(test_groups)
