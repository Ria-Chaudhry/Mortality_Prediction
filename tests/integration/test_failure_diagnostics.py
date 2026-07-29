from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
import yaml

from clinical_domain_mortality.config import PROJECT_ROOT, read_yaml
from clinical_domain_mortality.errors import CountMismatchError, IntegrityError
from clinical_domain_mortality.pipeline import (
    _enforce_expected_event_counts,
    _enforce_expected_selection_counts,
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
    comparison = pd.read_csv(diagnostic / "expected_vs_observed_counts.csv")
    assert {
        "category",
        "stage",
        "measure",
        "absolute_difference",
        "tolerance",
        "matches",
    } <= set(comparison)


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
    assert {
        "absolute_difference",
        "tolerance",
        "matches",
    } <= set(comparison)
    assert (tmp_path / "failed_run_manifest.json").is_file()


def test_selection_count_failure_preserves_fold_domain_comparison(
    tmp_path, chorus_config
):
    config = deepcopy(chorus_config)
    config["paper_run"] = True
    config["paper"] = {
        "expected_selection_counts": {
            "selected_concepts_per_fold_domain": 50,
            "selected_features_per_fold_domain": 21,
            "candidate_measurements": 300,
            "candidate_medications": 104,
            "candidate_procedures": 103,
        },
        "expected_count_tolerances": {"default": 0},
    }
    concept_rows = []
    derived_rows = []
    candidate_counts = {
        "measurements": 300,
        "medications": 104,
        "procedures": 103,
    }
    for fold in range(5):
        for domain, candidate_count in candidate_counts.items():
            concept_rows.extend(
                {"fold": fold, "domain": domain}
                for _ in range(50)
            )
            derived_rows.extend(
                {
                    "fold": fold,
                    "domain": domain,
                    "selected": rank <= (
                        20 if fold == 0 and domain == "measurements" else 21
                    ),
                }
                for rank in range(1, candidate_count + 1)
            )
    with pytest.raises(IntegrityError, match="selection-count mismatch"):
        _enforce_expected_selection_counts(
            config,
            pd.DataFrame(concept_rows),
            pd.DataFrame(derived_rows),
            tmp_path,
            "mapping-hash",
        )
    comparison = pd.read_csv(
        tmp_path / "expected_vs_observed_selection_counts.csv"
    )
    failed = comparison.loc[
        comparison["fold"].eq(0)
        & comparison["domain"].eq("measurements")
        & comparison["measure"].eq("selected_features")
    ].iloc[0]
    assert failed["observed"] == 20
    assert not bool(failed["matches"])
    assert (tmp_path / "failed_run_manifest.json").is_file()
