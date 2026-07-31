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


def read_table(
    path: str | Path,
    *,
    columns: list[str],
    dtypes: dict[str, str] | None = None,
    allowed_values: dict[str, set[Any]] | None = None,
    allowed_any: dict[str, set[Any]] | None = None,
    primary_or_fallback: tuple[str, set[Any], str, set[Any]] | None = None,
    time_bounds: tuple[str, pd.Timestamp, pd.Timestamp] | None = None,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Read selected columns with predicate pushdown or bounded CSV chunks."""
    resolved = Path(path)
    suffixes = resolved.suffixes
    if resolved.suffix == ".parquet":
        import pyarrow.dataset as ds

        dataset = ds.dataset(resolved, format="parquet")
        missing = set(columns) - set(dataset.schema.names)
        if missing:
            raise ConfigurationError(
                f"{resolved.name} missing required columns: {sorted(missing)}"
            )
        expression = None
        for name, values in sorted((allowed_values or {}).items()):
            condition = ds.field(name).isin(sorted(values, key=str))
            expression = condition if expression is None else expression & condition
        any_expression = None
        for name, values in sorted((allowed_any or {}).items()):
            condition = ds.field(name).isin(sorted(values, key=str))
            any_expression = (
                condition if any_expression is None else any_expression | condition
            )
        if any_expression is not None:
            expression = (
                any_expression if expression is None else expression & any_expression
            )
        if primary_or_fallback:
            primary, primary_values, fallback, fallback_values = (
                primary_or_fallback
            )
            condition = ds.field(primary).isin(
                sorted(primary_values, key=str)
            ) | (
                ds.field(primary).is_null()
                & ds.field(fallback).isin(
                    sorted(fallback_values, key=str)
                )
            )
            expression = condition if expression is None else expression & condition
        if time_bounds:
            name, lower, upper = time_bounds
            condition = (ds.field(name) >= lower.to_pydatetime()) & (
                ds.field(name) < upper.to_pydatetime()
            )
            expression = condition if expression is None else expression & condition
        frame = dataset.to_table(columns=columns, filter=expression).to_pandas()
        for name, dtype in (dtypes or {}).items():
            if name in frame:
                frame[name] = frame[name].astype(dtype)
        return frame
    if resolved.suffix == ".csv" or suffixes[-2:] == [".csv", ".gz"]:
        parts: list[pd.DataFrame] = []
        try:
            iterator = pd.read_csv(
                resolved,
                usecols=columns,
                dtype=dtypes,
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in iterator:
                keep = pd.Series(True, index=chunk.index)
                for name, values in (allowed_values or {}).items():
                    keep &= chunk[name].isin(values)
                if allowed_any:
                    any_keep = pd.Series(False, index=chunk.index)
                    for name, values in allowed_any.items():
                        any_keep |= chunk[name].isin(values)
                    keep &= any_keep
                if primary_or_fallback:
                    primary, primary_values, fallback, fallback_values = (
                        primary_or_fallback
                    )
                    keep &= chunk[primary].isin(primary_values) | (
                        chunk[primary].isna()
                        & chunk[fallback].isin(fallback_values)
                    )
                if time_bounds:
                    name, lower, upper = time_bounds
                    times = pd.to_datetime(chunk[name], errors="coerce")
                    keep &= times.ge(lower) & times.lt(upper)
                if keep.any():
                    parts.append(chunk.loc[keep].copy())
        except ValueError as error:
            raise ConfigurationError(
                f"{resolved.name} does not satisfy its required native columns: {error}"
            ) from error
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    raise ConfigurationError(f"Unsupported table format: {resolved}")


def write_csv(frame: pd.DataFrame, path: str | Path) -> str:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.10f")
    temporary.replace(resolved)
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
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(resolved)
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
