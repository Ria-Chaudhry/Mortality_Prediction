from __future__ import annotations

import argparse
import json

import pandas as pd

from clinical_domains.core.audit import dataframe_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    args = parser.parse_args()
    print(json.dumps(dataframe_audit(pd.read_csv(args.table)), indent=2))


if __name__ == "__main__":
    main()
