"""Compatibility entry point for the commands printed in the scientific specification."""

from __future__ import annotations

import argparse

from clinical_domain_mortality.cli import main


def compatibility_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-test", action="store_true")
    parser.add_argument("--verify-manifest")
    parser.add_argument("--dataset")
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.synthetic_test:
        return main(["synthetic-run"])
    if args.verify_manifest:
        from pathlib import Path

        return main(["verify", "--run-dir", str(Path(args.verify_manifest).parent)])
    if args.dataset and args.config:
        return main(["run", "--dataset", args.dataset, "--config", args.config])
    parser.error("Choose --synthetic-test, --verify-manifest, or --dataset with --config")
    return 2


if __name__ == "__main__":
    raise SystemExit(compatibility_main())
