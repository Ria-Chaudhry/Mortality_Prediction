from __future__ import annotations

import argparse
from pathlib import Path

from clinical_domains.adapters.generic_ehr import GenericEHRAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    adapter = GenericEHRAdapter.from_config(args.config)
    data = adapter.extract_all()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in data.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    print(f"Wrote standardized adapter outputs to {output_dir}")


if __name__ == "__main__":
    main()
