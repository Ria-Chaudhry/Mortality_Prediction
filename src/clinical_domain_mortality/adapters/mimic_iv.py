"""MIMIC-IV local-file adapter."""

from __future__ import annotations

from .base import LocalFileAdapter, StandardizedData


class MIMICIVAdapter(LocalFileAdapter):
    """Normalize user-supplied MIMIC-IV CSV, CSV.GZ, or Parquet tables."""

    def load(self) -> StandardizedData:
        raw = self._load_local_tables()
        result = self._build_result(raw)
        result.audit["mimic_expected_version"] = self.source.get("expected_version")
        result.audit["medication_source_semantics"] = self.source.get("source_semantics", {}).get(
            "medications", {}
        )
        result.audit["procedure_source_semantics"] = self.source.get("source_semantics", {}).get(
            "procedures", {}
        )
        return result
