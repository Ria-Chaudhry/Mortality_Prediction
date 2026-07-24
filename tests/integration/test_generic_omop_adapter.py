import pytest

from clinical_domains.adapters.generic_omop import GenericOMOPAdapter


def test_generic_omop_adapter_is_template_until_connected():
    adapter = GenericOMOPAdapter({})
    with pytest.raises(NotImplementedError):
        adapter.extract_encounters()
