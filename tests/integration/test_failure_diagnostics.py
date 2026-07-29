from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
import yaml

from clinical_domain_mortality.config import PROJECT_ROOT, read_yaml
from clinical_domain_mortality.errors import CountMismatchError, IntegrityError
from clinical_domain_mortality.pipeline import (
    _enforce_expected_event_counts,
    run_pipeline,
)


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


def test_event_count_failure_preserves_observed_comparison(tmp_path, chorus_config):
    config = deepcopy(chorus_config)
    config["paper_run"] = True
    config["paper"] = {
        "expected_event_counts": {
            "measurements": 999,
            "medications.qualifying": 2,
        }
    }
    audit = pd.DataFrame(
        [
            {"domain": "measurements", "status": "qualifying", "count": 4},
            {"domain": "medications", "status": "qualifying", "count": 2},
        ]
    )
    with pytest.raises(IntegrityError, match="event-count mismatch"):
        _enforce_expected_event_counts(config, audit, tmp_path, "mapping-hash")
    comparison = pd.read_csv(tmp_path / "expected_vs_observed_event_counts.csv")
    assert comparison.set_index("domain").loc["measurements", "observed"] == 4
    assert (tmp_path / "failed_run_manifest.json").is_file()
