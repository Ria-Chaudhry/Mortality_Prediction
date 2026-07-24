from clinical_domains.evaluation.decision_curve import net_benefit


def test_net_benefit_returns_float():
    assert isinstance(net_benefit([0, 1], [0.1, 0.9], 0.5), float)
