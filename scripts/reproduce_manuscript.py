from __future__ import annotations

import argparse
from pathlib import Path

from clinical_domains.pipeline import run_reproduction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_reproduction(args.config, args.output_dir)
    output_dir = Path(args.output_dir)
    print(f"Wrote synthetic reproduction outputs to {output_dir}")


if __name__ == "__main__":
    main()
