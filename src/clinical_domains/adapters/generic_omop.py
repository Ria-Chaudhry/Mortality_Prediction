from __future__ import annotations

from clinical_domains.adapters.base import DatasetAdapter


class GenericOMOPAdapter(DatasetAdapter):
    """Template adapter for OMOP CDM sites.

    Site-specific SQL templates and concept mappings should live outside the analytical
    core in `adapters/generic_omop/`.
    """

    def __init__(self, config: dict):
        self.config = config

    def extract_encounters(self):
        raise NotImplementedError("Connect site SQL templates before using this adapter.")

    def extract_baseline(self):
        raise NotImplementedError("Connect site SQL templates before using this adapter.")

    def extract_events(self):
        raise NotImplementedError("Connect site SQL templates before using this adapter.")

    def extract_mortality(self):
        raise NotImplementedError("Connect site SQL templates before using this adapter.")
