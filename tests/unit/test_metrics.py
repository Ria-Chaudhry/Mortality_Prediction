from clinical_domains.evaluation.discrimination import auroc, average_precision


def test_discrimination_metrics_are_bounded():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]

    assert 0 <= auroc(y_true, y_score) <= 1
    assert 0 <= average_precision(y_true, y_score) <= 1
