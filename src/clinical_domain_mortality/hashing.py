"""Canonical SHA-256 hashing for configs, rows, files, and manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, pd.Timestamp | np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot hash value of type {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
        allow_nan=False,
    )


def hash_object(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_files(paths: Iterable[str | Path]) -> str:
    records = [
        {"path": str(Path(path).as_posix()), "sha256": hash_file(path)}
        for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix())
    ]
    return hash_object(records)


def hash_frame(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame.loc[:, columns] if columns else frame
    records = selected.astype(object).where(pd.notna(selected), None).to_dict(orient="records")
    return hash_object(records)


def hash_frame_schema(frame: pd.DataFrame) -> str:
    """Hash ordered column names and dtypes, without hashing values."""
    return hash_object(
        [{"name": str(column), "dtype": str(frame[column].dtype)} for column in frame]
    )


def hash_frame_values(
    frame: pd.DataFrame,
    *,
    identity_columns: list[str] | None = None,
) -> str:
    """Hash ordered identities, columns, dtypes, and values."""
    if identity_columns:
        missing = set(identity_columns) - set(frame)
        if missing:
            raise ValueError(f"Missing identity columns for feature hash: {sorted(missing)}")
    return hash_object(
        {
            "schema": [
                {"name": str(column), "dtype": str(frame[column].dtype)}
                for column in frame
            ],
            "identity_columns": identity_columns or [],
            "records": frame.astype(object)
            .where(pd.notna(frame), None)
            .to_dict(orient="records"),
        }
    )


def hash_frame_canonical(frame: pd.DataFrame) -> str:
    """Content-sensitive, row-order-independent digest of an analytical table."""
    schema = sorted(
        [{"name": str(column), "dtype": str(frame[column].dtype)} for column in frame],
        key=lambda item: item["name"],
    )
    ordered_columns = [item["name"] for item in schema]
    records = (
        frame.loc[:, ordered_columns]
        .astype(object)
        .where(pd.notna(frame.loc[:, ordered_columns]), None)
        .to_dict(orient="records")
    )
    row_hashes = sorted(hash_object(record) for record in records)
    return hash_object({"schema": schema, "rows": row_hashes, "row_count": len(records)})
