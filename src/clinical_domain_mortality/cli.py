"""Single public command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .adapters import CHoRUSAdapter, MIMICIVAdapter
from .config import load_config
from .errors import PipelineError
from .pipeline import (
    freeze_synthetic_expected,
    run_pipeline,
    synthetic_run,
    verify_paper_run,
    verify_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clinical-domain-mortality")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate configuration and source mapping")
    validate.add_argument("--config", required=True, type=Path)

    run = subparsers.add_parser("run", help="Execute one independent dataset analysis")
    run.add_argument("--dataset", choices=["chorus", "mimic", "mimiciv"])
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--restricted-output-dir", type=Path)

    synthetic = subparsers.add_parser(
        "synthetic-run", help="Execute both public synthetic adapters"
    )
    synthetic.set_defaults(command="synthetic-run")

    verify = subparsers.add_parser("verify", help="Verify checksums and output invariants")
    verify.add_argument("--run-dir", required=True, type=Path)

    freeze = subparsers.add_parser(
        "freeze-synthetic-expected",
        help="Intentionally update all deterministic synthetic expected hashes",
    )
    freeze.add_argument("--run-dir", required=True, type=Path)
    freeze.add_argument("--approve-update", action="store_true")

    verify_paper = subparsers.add_parser(
        "verify-paper", help="Fail-closed verification of an executed paper run"
    )
    verify_paper.add_argument("--config", required=True, type=Path)
    verify_paper.add_argument("--run-dir", required=True, type=Path)

    stage = subparsers.add_parser("stage", help="Run through a numbered auditable stage")
    stage.add_argument("--stage", required=True, type=int, choices=range(1, 9))
    stage.add_argument("--config", required=True, type=Path)
    stage.add_argument("--output-dir", type=Path)
    stage.add_argument("--restricted-output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            config = load_config(args.config)
            adapter = (
                CHoRUSAdapter(config)
                if config["adapter"] == "chorus"
                else MIMICIVAdapter(config)
            )
            result = adapter.load()
            payload = {
                "status": "ok",
                "dataset": config["dataset"],
                "adapter": config["adapter"],
                "config_hash": config["_meta"]["config_hash"],
                "mapping_hash": result.mapping_hash,
                "rows": {name: len(frame) for name, frame in result.tables.items()},
            }
        elif args.command == "run":
            config = load_config(args.config)
            requested = "mimiciv" if args.dataset == "mimic" else args.dataset
            if requested and requested != config["dataset"]:
                raise PipelineError(
                    f"--dataset {args.dataset!r} disagrees with config dataset {config['dataset']!r}"
                )
            result = run_pipeline(
                args.config,
                args.output_dir,
                args.restricted_output_dir,
            )
            payload = {
                "status": "ok",
                "dataset": result.dataset,
                "public_dir": str(result.public_dir),
                "run_id": result.run_manifest["run_id"],
            }
        elif args.command == "synthetic-run":
            payload = {"status": "ok", **synthetic_run()}
        elif args.command == "verify":
            payload = verify_run(args.run_dir)
        elif args.command == "freeze-synthetic-expected":
            payload = freeze_synthetic_expected(
                args.run_dir, approve_update=args.approve_update
            )
        elif args.command == "verify-paper":
            payload = verify_paper_run(args.config, args.run_dir)
        elif args.command == "stage":
            result = run_pipeline(
                args.config,
                args.output_dir,
                args.restricted_output_dir,
                stop_after=args.stage,
            )
            payload = {
                "status": "ok",
                "dataset": result.dataset,
                "stage": args.stage,
                "public_dir": str(result.public_dir),
            }
        else:
            parser.error("Unknown command")
            return 2
    except PipelineError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
