"""Publish bounded CI failure details as GitHub check annotations."""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _escape(message: str) -> str:
    return (
        message.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def _emit(title: str, message: str) -> None:
    bounded = message.strip()[-6000:] or "No diagnostic text was recorded."
    print(f"::error title={_escape(title)}::{_escape(bounded)}")


def report_junit(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    for case in ET.parse(path).iter("testcase"):
        failure = case.find("failure")
        error = case.find("error")
        detail = failure if failure is not None else error
        if detail is None:
            continue
        identity = ".".join(
            value
            for value in (case.get("classname"), case.get("name"))
            if value
        )
        _emit(f"pytest: {identity}", detail.text or detail.get("message", ""))
        count += 1
    return count


def report_log(path: Path) -> int:
    if not path.is_file():
        return 0
    contents = path.read_text(encoding="utf-8").strip()
    if not contents or '"status": "failed"' not in contents:
        return 0
    _emit("synthetic verification", contents)
    return 1


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def report_synthetic_comparison(root: Path) -> int:
    reported = 0
    metric_fields = ("matrix", "model", "auprc", "auroc", "brier", "log_loss")
    for dataset in ("chorus", "mimiciv"):
        dataset_dir = root / dataset
        best = _read_csv(dataset_dir / "best_model_by_matrix.csv")
        pooled = [
            {field: row.get(field, "") for field in metric_fields}
            for row in _read_csv(dataset_dir / "pooled_oof_metrics.csv")
            if row.get("matrix") in {"baseline", "baseline_all_domains"}
        ]
        if not best and not pooled:
            continue
        compact_best = [
            {
                field: row.get(field, "")
                for field in ("matrix", "model", "auprc", "auroc", "brier")
            }
            for row in best
        ]
        _emit(
            f"Linux synthetic comparison: {dataset}",
            json.dumps(
                {"best_models": compact_best, "pooled_reference": pooled},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        reported += 1
    return reported


def report_observed_freeze_hashes(root: Path) -> int:
    from clinical_domain_mortality.hashing import hash_object
    from clinical_domain_mortality.io import read_json
    from clinical_domain_mortality.pipeline import (
        PROJECT_ROOT,
        _canonical_artifact_hash,
        _safe_parent_manifest,
        _safe_run_manifest,
    )

    expected = read_json(
        PROJECT_ROOT
        / "synthetic_data"
        / "expected_outputs"
        / "expected_aggregate_hashes.json"
    )
    candidate = {
        key: expected[key]
        for key in (
            "format_version",
            "canonical_float_decimal_places",
            "reference_runtime",
        )
    }
    candidate["datasets"] = {}
    reported = 0
    for dataset in ("chorus", "mimiciv"):
        dataset_dir = root / dataset
        frozen = expected["datasets"][dataset]
        actual_names = sorted(
            path.relative_to(dataset_dir).as_posix()
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.name != "run_manifest.json"
        )
        observed = {
            relative: _canonical_artifact_hash(dataset_dir / relative)
            for relative in actual_names
        }
        changed = {
            relative: digest
            for relative, digest in observed.items()
            if digest != frozen["artifact_hashes"].get(relative)
        }
        payload = {
            "artifact_names": actual_names,
            "changed_artifact_hashes": changed,
            "safe_run_manifest_hash": hash_object(
                _safe_run_manifest(read_json(dataset_dir / "run_manifest.json"))
            ),
        }
        candidate["datasets"][dataset] = {
            "artifact_names": actual_names,
            "artifact_hashes": observed,
            "safe_run_manifest_hash": payload["safe_run_manifest_hash"],
        }
        _emit(
            f"Ubuntu freeze hashes: {dataset}",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        reported += 1
    candidate["safe_parent_manifest_hash"] = hash_object(
        _safe_parent_manifest(read_json(root / "run_manifest.json"))
    )
    candidate["synthetic_run_summary_hash"] = _canonical_artifact_hash(
        root / "synthetic_run_summary.csv"
    )
    candidate_path = Path("outputs/test-reports/observed_freeze_candidate.json")
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _emit(
        "Ubuntu freeze hashes: parent",
        json.dumps(
            {
                "safe_parent_manifest_hash": candidate[
                    "safe_parent_manifest_hash"
                ],
                "synthetic_run_summary_hash": candidate[
                    "synthetic_run_summary_hash"
                ],
                "complete_candidate": candidate_path.as_posix(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return reported + 1


def main() -> int:
    report_dir = Path("outputs/test-reports")
    reported = report_junit(report_dir / "pytest-full.xml")
    verify_failed = report_log(report_dir / "verify.log")
    reported += verify_failed
    if verify_failed:
        reported += report_synthetic_comparison(Path("outputs/synthetic"))
        reported += report_observed_freeze_hashes(Path("outputs/synthetic"))
    if reported == 0:
        _emit("CI failure", "A prior step failed without a diagnostic report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
