from pathlib import Path

from clinical_domains.adapters.generic_ehr import GenericEHRAdapter


def test_generic_ehr_adapter_reads_synthetic_data():
    adapter = GenericEHRAdapter.from_config(Path("examples/synthetic/config.yaml"))
    data = adapter.extract_all()

    assert set(data) == {"encounters", "baseline", "events", "mortality"}
    assert len(data["events"]) > 0
