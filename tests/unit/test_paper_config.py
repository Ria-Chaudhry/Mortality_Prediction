from __future__ import annotations

import pytest

from clinical_domain_mortality.config import PROJECT_ROOT, load_config
from clinical_domain_mortality.errors import ConfigurationError


@pytest.mark.parametrize(
    "name", ["chorus.paper.yaml", "mimiciv.paper.yaml"]
)
def test_unconfirmed_paper_configs_fail_closed(name):
    with pytest.raises(ConfigurationError, match="fail-closed"):
        load_config(PROJECT_ROOT / "configs" / name)
