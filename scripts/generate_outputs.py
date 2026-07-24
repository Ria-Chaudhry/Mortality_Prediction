from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clinical_domains.reporting.manuscript_outputs import write_manuscript_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    path = write_manuscript_tables(pd.read_csv(args.metrics), Path(args.output_dir))
    print(path)


if __name__ == "__main__":
    main()
