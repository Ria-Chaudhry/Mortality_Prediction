"""Deterministic table and manifest input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import ConfigurationError, IntegrityError
from .hashing import hash_file


def find_table(root: Path, name: str, format_hint: str = "auto") -> Path:
    direct = root / name
    candidates = [direct] if direct.suffix else []
    if format_hint in {"auto", "csv"}:
        candidates.extend([root / f"{name}.csv", root / f"{name}.csv.gz"])
    if format_hint in {"auto", "parquet"}:
        candidates.append(root / f"{name}.parquet")
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ConfigurationError(
            f"Expected exactly one source file for {name!r}; found {[str(path) for path in found]}"
        )
    return found[0]


def read_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    suffixes = resolved.suffixes
    if resolved.suffix == ".parquet":
        return pd.read_parquet(resolved)
    if resolved.suffix == ".csv" or suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(resolved, low_memory=False)
    raise ConfigurationError(f"Unsupported table format: {resolved}")


def write_csv(frame: pd.DataFrame, path: str | Path) -> str:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolved, index=False, lineterminator="\n", float_format="%.17g")
    return hash_file(resolved)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer | np.floating):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp | np.datetime64):
        return pd.Timestamp(value).isoformat()
    return value


def write_json(value: Any, path: str | Path) -> str:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_sanitize(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    resolved.write_text(payload, encoding="utf-8")
    return hash_file(resolved)


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_hashes(base: Path, hashes: dict[str, str]) -> None:
    failures = []
    for relative, expected in sorted(hashes.items()):
        target = base / relative
        actual = hash_file(target) if target.is_file() else None
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    if failures:
        raise IntegrityError(f"Checksum verification failed: {failures}")
