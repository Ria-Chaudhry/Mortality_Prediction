from __future__ import annotations

import argparse

import pandas as pd

from clinical_domains.reporting.tables import metric_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain-set", default="all_domains")
    args = parser.parse_args()
    metrics = metric_table(pd.read_csv(args.predictions), domain_set=args.domain_set)
    metrics.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
