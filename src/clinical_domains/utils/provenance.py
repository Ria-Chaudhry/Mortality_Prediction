from __future__ import annotations

from datetime import datetime, timezone


def run_metadata(config_path: str | None = None) -> dict[str, str | None]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
    }
