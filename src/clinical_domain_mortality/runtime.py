"""Supported runtime guard for byte-stable frozen synthetic verification."""

from __future__ import annotations

import platform

from .errors import ConfigurationError

SUPPORTED_PYTHON = "3.10.13"
FROZEN_REFERENCE_SYSTEM = "Linux"
FROZEN_REFERENCE_MACHINE = "x86_64"


def validate_supported_runtime() -> None:
    """Fail clearly outside the sole Python runtime used for execution."""
    observed = platform.python_version()
    if observed != SUPPORTED_PYTHON:
        raise ConfigurationError(
            "Frozen execution supports CPython "
            f"{SUPPORTED_PYTHON}; observed {observed}. Use the pinned runtime."
        )


def frozen_verification_runtime_supported() -> bool:
    """Return whether exact fitted-model pins apply on this runtime."""
    return (
        platform.python_version() == SUPPORTED_PYTHON
        and platform.system() == FROZEN_REFERENCE_SYSTEM
        and platform.machine() == FROZEN_REFERENCE_MACHINE
    )


def validate_frozen_verification_runtime() -> None:
    """Fail closed outside the frozen Ubuntu-compatible reference platform."""
    validate_supported_runtime()
    observed = f"{platform.system()} {platform.machine()}"
    if not frozen_verification_runtime_supported():
        raise ConfigurationError(
            "Exact frozen fitted-model verification supports only "
            f"{FROZEN_REFERENCE_SYSTEM} {FROZEN_REFERENCE_MACHINE} with "
            f"CPython {SUPPORTED_PYTHON}; observed {observed}. Synthetic "
            "execution and same-runtime repeat checks remain available, but "
            "this runtime cannot certify the Ubuntu reference artifacts."
        )
