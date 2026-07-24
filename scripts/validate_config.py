from __future__ import annotations

import argparse
from pathlib import Path

from clinical_domains.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = Path(args.config)
    config = load_yaml(path)
    if "dataset" not in config and "paths" not in config:
        raise SystemExit("Config should include at least dataset metadata or paths.")
    print(f"Config OK: {path}")


if __name__ == "__main__":
    main()
