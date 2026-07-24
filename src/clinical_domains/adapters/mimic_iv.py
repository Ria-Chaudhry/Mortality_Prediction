from __future__ import annotations

from clinical_domains.adapters.base import DatasetAdapter


class MimicIVAdapter(DatasetAdapter):
    """MIMIC-IV adapter placeholder.

    MIMIC-IV extraction SQL and item mappings live in `adapters/mimic_iv/`.
    """

    def __init__(self, config: dict):
        self.config = config

    def extract_encounters(self):
        raise NotImplementedError("Implement MIMIC-IV admission extraction.")

    def extract_baseline(self):
        raise NotImplementedError("Implement MIMIC-IV baseline extraction.")

    def extract_events(self):
        raise NotImplementedError("Implement MIMIC-IV event extraction.")

    def extract_mortality(self):
        raise NotImplementedError("Implement MIMIC-IV mortality extraction.")
