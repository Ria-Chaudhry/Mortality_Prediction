import pandas as pd

from clinical_domains.features.selection import top_variance_features


def test_top_variance_features_returns_highest_variance_columns():
    frame = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": ["x", "y", "z"]})

    assert top_variance_features(frame, max_features=1) == ["b"]
