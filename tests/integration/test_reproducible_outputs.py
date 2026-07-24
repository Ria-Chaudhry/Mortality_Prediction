from clinical_domains.utils.seeds import set_global_seed


def test_seed_helper_is_reproducible():
    import numpy as np

    set_global_seed(123)
    first = np.random.random()
    set_global_seed(123)
    second = np.random.random()
    assert first == second
