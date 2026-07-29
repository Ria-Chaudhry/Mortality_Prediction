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


def main() -> int:
    report_dir = Path("outputs/test-reports")
    reported = report_junit(report_dir / "pytest-full.xml")
    verify_failed = report_log(report_dir / "verify.log")
    reported += verify_failed
    if verify_failed:
        reported += report_synthetic_comparison(Path("outputs/synthetic"))
    if reported == 0:
        _emit("CI failure", "A prior step failed without a diagnostic report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
