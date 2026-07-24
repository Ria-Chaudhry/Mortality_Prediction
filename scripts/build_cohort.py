from __future__ import annotations

import argparse

import pandas as pd

from clinical_domains.core.cohort import build_cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encounters", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-age", type=int, default=18)
    args = parser.parse_args()
    cohort = build_cohort(pd.read_csv(args.encounters), min_age=args.min_age)
    cohort.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
