from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clinical_domains.adapters.generic_ehr import GenericEHRAdapter
from clinical_domains.core.cohort import build_cohort
from clinical_domains.core.landmark import restrict_events_to_landmark
from clinical_domains.core.outcomes import build_mortality_labels
from clinical_domains.features.aggregation import aggregate_events
from clinical_domains.features.matrices import build_feature_matrix, select_domain_columns
from clinical_domains.modeling.model_selection import cross_validated_predictions
from clinical_domains.reporting.manuscript_outputs import write_manuscript_tables
from clinical_domains.reporting.tables import metric_table
from clinical_domains.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    landmark_hours = config.get("study", {}).get("landmark_hours", 24)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = GenericEHRAdapter.from_config(args.config).extract_all()
    cohort = build_cohort(data["encounters"], min_age=config.get("study", {}).get("min_age", 18))
    landmarked = restrict_events_to_landmark(data["events"], cohort, hours=landmark_hours)
    event_features = aggregate_events(landmarked)
    matrix = build_feature_matrix(cohort, data["baseline"], event_features)
    labels = build_mortality_labels(cohort, data["mortality"]).rename(columns={"died": "outcome"})
    matrix = matrix.merge(labels[["encounter_id", "outcome"]], on="encounter_id", how="inner")

    feature_columns = select_domain_columns(
        matrix, ["baseline", "physiological", "treatment", "procedures"]
    )
    predictions = cross_validated_predictions(
        matrix,
        feature_columns,
        n_splits=config.get("modeling", {}).get("cv", {}).get("n_splits", 3),
    )
    metrics = metric_table(predictions)

    matrix.to_csv(output_dir / "feature_matrix.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    write_manuscript_tables(metrics, output_dir)
    print(f"Wrote synthetic reproduction outputs to {output_dir}")


if __name__ == "__main__":
    main()
