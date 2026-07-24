from __future__ import annotations

from pathlib import Path

import pandas as pd

from clinical_domains.adapters.base import DatasetAdapter
from clinical_domains.utils.config import load_yaml


class GenericEHRAdapter(DatasetAdapter):
    """CSV-backed adapter for already-harmonized non-OMOP EHR extracts."""

    def __init__(self, config: dict, config_dir: Path | None = None):
        self.config = config
        self.config_dir = config_dir or Path.cwd()

    @classmethod
    def from_config(cls, path: str | Path) -> GenericEHRAdapter:
        config_path = Path(path)
        return cls(load_yaml(config_path), config_path.parent)

    def _resolve(self, key: str) -> Path:
        raw = Path(self.config["paths"][key])
        return raw if raw.is_absolute() else (self.config_dir / raw).resolve()

    def _read(self, key: str) -> pd.DataFrame:
        return pd.read_csv(self._resolve(key))

    def extract_encounters(self) -> pd.DataFrame:
        frame = self._read("encounters")
        for column in ["admit_time", "discharge_time"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])
        return frame

    def extract_baseline(self) -> pd.DataFrame:
        return self._read("baseline")

    def extract_events(self) -> pd.DataFrame:
        frame = self._read("events")
        if "event_time" in frame.columns:
            frame["event_time"] = pd.to_datetime(frame["event_time"])
        return frame

    def extract_mortality(self) -> pd.DataFrame:
        frame = self._read("mortality")
        if "death_time" in frame.columns:
            frame["death_time"] = pd.to_datetime(frame["death_time"])
        return frame
