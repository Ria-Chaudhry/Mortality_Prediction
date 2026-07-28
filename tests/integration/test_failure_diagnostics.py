from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from clinical_domain_mortality.config import PROJECT_ROOT, read_yaml
from clinical_domain_mortality.errors import CountMismatchError
from clinical_domain_mortality.pipeline import run_pipeline


def test_count_failure_preserves_attrition_and_comparison(tmp_path):
    source = deepcopy(read_yaml(PROJECT_ROOT / "configs" / "chorus.example.yaml"))
    source["paper_run"] = False
    source["synthetic"] = False
    source["source"]["release_or_snapshot"] = "synthetic-diagnostic-fixture"
    source["overrides"]["cohort"] = {
        "expected_counts": {"visits": 1, "patients": 1, "deaths": 1},
        "enforce_expected_counts": True,
    }
    config_path = tmp_path / "count_failure.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    restricted = tmp_path / "restricted"
    with pytest.raises(CountMismatchError):
        run_pipeline(
            config_path,
            output_base=tmp_path / "public",
            restricted_base=restricted,
        )
    diagnostic = restricted / "chorus" / "release_candidate_aggregate"
    assert (diagnostic / "attrition.csv").is_file()
    assert (diagnostic / "expected_vs_observed_counts.csv").is_file()
    assert (diagnostic / "failed_run_manifest.json").is_file()
