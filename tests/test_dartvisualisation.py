import numpy as np
import pytest

from benchmarks import dartvisualisation as visualisation


def test_logistic_ratios_include_all_finite_rows():
    rows = []
    for seed, dartValue, malaValue in ((1, 2.0, 1.0), (2, 0.01, 1.0)):
        rows.extend((
            {
                'analysis': 'logistic_comparison',
                'method': 'dart_laplace',
                'evaluation_seed': seed,
                'dimension': 4,
                'slow_direction_ess_per_iteration': dartValue,
            },
            {
                'analysis': 'logistic_comparison',
                'method': 'mala',
                'evaluation_seed': seed,
                'dimension': 4,
                'slow_direction_ess_per_iteration': malaValue,
            },
        ))

    ratios = visualisation.logistic_ratios(rows)

    assert ratios[4].tolist() == [2.0, 0.01]


def test_localisation_ratios_use_ratios_of_means():
    rows = []
    for seed, localised, unlocalised in ((1, 4.0, 2.0), (2, 8.0, 2.0)):
        rows.extend((
            {
                'analysis': 'localisation_ablation',
                'method': 'localised_dart',
                'evaluation_seed': seed,
                'dimension': 8,
                'surrogate_condition': 'laplace',
                visualisation.EFFICIENCY_FIELD: localised,
            },
            {
                'analysis': 'localisation_ablation',
                'method': 'unlocalised_surrogate',
                'evaluation_seed': seed,
                'dimension': 8,
                'surrogate_condition': 'laplace',
                visualisation.EFFICIENCY_FIELD: unlocalised,
            },
        ))

    ratios = visualisation.localisation_ratios(rows)

    assert ratios[(8, 'laplace')] == pytest.approx(3.0)


def test_localisation_trend_fits_shared_power_law_slope():
    exponent = np.log2(1.5)
    ratios = {
        (dimension, condition): intercept * dimension ** exponent
        for dimension in (4, 8, 12, 16)
        for condition, intercept in (('laplace', 0.5), ('shift', 0.8))
    }

    _, _, multiplier = visualisation.localisation_trend(ratios)

    assert multiplier == pytest.approx(1.5)
