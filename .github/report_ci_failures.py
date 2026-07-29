"""Publish bounded CI failure details as GitHub check annotations."""

from __future__ import annotations

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
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return 0
    _emit("synthetic verification", path.read_text(encoding="utf-8"))
    return 1


def main() -> int:
    report_dir = Path("outputs/test-reports")
    reported = report_junit(report_dir / "pytest-full.xml")
    reported += report_log(report_dir / "verify.log")
    if reported == 0:
        _emit("CI failure", "A prior step failed without a diagnostic report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
