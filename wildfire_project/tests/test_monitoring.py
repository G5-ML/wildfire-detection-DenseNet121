import numpy as np

from wildfire_ml.monitoring import population_stability_index


def test_population_stability_is_zero_for_identical_samples() -> None:
    scores = np.linspace(0.01, 0.99, 100)

    assert population_stability_index(scores, scores) == 0.0


def test_population_stability_detects_large_shift() -> None:
    reference = np.linspace(0.01, 0.3, 200)
    shifted = np.linspace(0.7, 0.99, 200)

    assert population_stability_index(reference, shifted) > 0.2
