import pytest

from clinical_domains.evaluation.shap_analysis import require_shap


def test_require_shap_is_callable():
    try:
        require_shap()
    except ImportError:
        pytest.skip("Optional SHAP dependency is not installed.")
