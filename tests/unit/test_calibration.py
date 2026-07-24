from clinical_domains.evaluation.calibration import brier_score


def test_brier_score_is_nonnegative():
    assert brier_score([0, 1], [0.2, 0.8]) >= 0
