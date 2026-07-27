"""Run provenance fields."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from ..config import PROJECT_ROOT


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_is_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None
