from __future__ import annotations

import pandas as pd
import pytest

from clinical_domain_mortality.config import PROJECT_ROOT, load_config
from clinical_domain_mortality.errors import ConfigurationError, IntegrityError
from clinical_domain_mortality.pipeline import _verify_selection_artifacts
from clinical_domain_mortality.runtime import validate_supported_runtime


@pytest.mark.parametrize(
    "name", ["chorus.paper.yaml", "mimiciv.paper.yaml"]
)
def test_unconfirmed_paper_configs_fail_closed(name):
    with pytest.raises(ConfigurationError, match="fail-closed"):
        load_config(PROJECT_ROOT / "configs" / name)


def test_unsupported_python_runtime_fails_before_execution(monkeypatch):
    monkeypatch.setattr("platform.python_version", lambda: "3.12.13")
    with pytest.raises(ConfigurationError, match="CPython 3.10.13"):
        validate_supported_runtime()


def test_paper_selection_verification_reads_actual_tables(tmp_path, chorus_config):
    concept_rows = []
    derived_rows = []
    for fold in range(5):
        fold_dir = tmp_path / "fold_features" / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        for domain in ("measurements", "medications", "procedures"):
            constructed = chorus_config["features"][domain]["constructed_count"]
            concept_rows.extend(
                {
                    "fold": fold,
                    "domain": domain,
                    "rank": rank,
                    "selected": True,
                }
                for rank in range(1, 51)
            )
            rows = []
            for rank in range(1, constructed + 1):
                rows.append(
                    {
                        "fold": fold,
                        "domain": domain,
                        "candidate_feature_name": f"{domain}_{rank}",
                        "source_concept": f"c{(rank - 1) // 6}",
                        "summary_type": "count",
                        "training_support_count": constructed - rank,
                        "training_support_proportion": (constructed - rank)
                        / constructed,
                        "selection_score": (constructed - rank) / constructed,
                        "tie_break_value": rank,
                        "rank": rank,
                        "selected": rank <= 21,
                        "selection_rule_identifier": "training_support_prevalence_v1",
                        "selection_rule_version": 1,
                    }
                )
            derived_rows.extend(rows)
            pd.DataFrame(
                {
                    "cohort_visit_number": [1],
                    **{f"{domain}_{rank}": [0] for rank in range(1, 22)},
                }
            ).to_parquet(fold_dir / f"{domain}.parquet", index=False)
    pd.DataFrame(concept_rows).to_csv(
        tmp_path / "fold_concept_selections.csv", index=False
    )
    derived_path = tmp_path / "fold_derived_feature_selections.csv"
    pd.DataFrame(derived_rows).to_csv(derived_path, index=False)
    _verify_selection_artifacts(chorus_config, tmp_path)
    corrupted = pd.read_csv(derived_path)
    corrupted.loc[
        (corrupted["fold"] == 0)
        & (corrupted["domain"] == "measurements")
        & (corrupted["rank"] == 21),
        "selected",
    ] = False
    corrupted.to_csv(derived_path, index=False)
    with pytest.raises(IntegrityError, match="exactly 21"):
        _verify_selection_artifacts(chorus_config, tmp_path)
