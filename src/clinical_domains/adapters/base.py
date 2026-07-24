from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

STANDARD_SCHEMAS = {
    "encounters": ["encounter_id", "patient_id", "admit_time", "discharge_time"],
    "baseline": ["encounter_id"],
    "events": ["encounter_id", "event_time", "domain", "feature_name", "value"],
    "mortality": ["encounter_id", "died"],
}


class DatasetAdapter(ABC):
    """Contract every dataset adapter must satisfy."""

    @abstractmethod
    def extract_encounters(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def extract_baseline(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def extract_events(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def extract_mortality(self) -> pd.DataFrame:
        raise NotImplementedError

    def extract_all(self) -> dict[str, pd.DataFrame]:
        data = {
            "encounters": self.extract_encounters(),
            "baseline": self.extract_baseline(),
            "events": self.extract_events(),
            "mortality": self.extract_mortality(),
        }
        for name, frame in data.items():
            self.validate_frame(name, frame)
        return data

    @staticmethod
    def validate_frame(name: str, frame: pd.DataFrame) -> None:
        required = set(STANDARD_SCHEMAS[name])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")
