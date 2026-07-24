from __future__ import annotations

import argparse

import pandas as pd

from clinical_domains.features.matrices import build_feature_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encounters", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--event-features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    matrix = build_feature_matrix(
        pd.read_csv(args.encounters),
        pd.read_csv(args.baseline),
        pd.read_csv(args.event_features),
    )
    matrix.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
