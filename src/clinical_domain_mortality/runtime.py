"""Supported runtime guard for byte-stable frozen synthetic verification."""

from __future__ import annotations

import platform

from .errors import ConfigurationError

SUPPORTED_PYTHON = "3.10.13"


def validate_supported_runtime() -> None:
    """Fail clearly outside the sole runtime used for frozen verification."""
    observed = platform.python_version()
    if observed != SUPPORTED_PYTHON:
        raise ConfigurationError(
            "Frozen execution supports CPython "
            f"{SUPPORTED_PYTHON}; observed {observed}. Use the pinned runtime."
        )
