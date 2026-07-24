from __future__ import annotations

import argparse

import pandas as pd

from clinical_domains.features.matrices import select_domain_columns
from clinical_domains.modeling.model_selection import cross_validated_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domains", nargs="+", default=["baseline", "physiological", "treatment", "procedures"])
    args = parser.parse_args()
    matrix = pd.read_csv(args.matrix)
    features = select_domain_columns(matrix, args.domains)
    predictions = cross_validated_predictions(matrix, features)
    predictions.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
