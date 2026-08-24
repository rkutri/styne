from dataclasses import replace

import pytest

from benchmarks import dartbenchmark as common
from benchmarks import dartlogistic as benchmark
from styne.backend import get_backend


METHODS = (common.DART, common.MALA, common.MLDA)


def test_production_protocol_is_fixed():
    configuration = benchmark.PRODUCTION_CONFIGURATION

    assert configuration.dimensions == (2, 4, 8, 16)
    assert configuration.evaluationSeeds == tuple(
        range(33_000, 35_000, 100)
    )
    assert len(configuration.dartGammaCandidates) == 25
    assert configuration.dartGammaCandidates[0] == pytest.approx(1e-5)
    assert configuration.reportingBurnin == 25_000
    assert configuration.reportingRetained == 100_000

    targets = len(configuration.dimensions) * len(
        configuration.evaluationSeeds
    )
    pilotRows = targets * sum(
        len(benchmark.pilot_candidates(configuration, method))
        for method in METHODS
    )
    assert pilotRows + targets * len(METHODS) == 5_600


def test_comparison_seeds_are_reproducible_and_method_specific():
    first = benchmark.comparison_seeds(31_000, 4)
    repeated = benchmark.comparison_seeds(31_000, 4)
    dart = benchmark.comparison_seeds(31_000, 4, common.DART)
    mala = benchmark.comparison_seeds(31_000, 4, common.MALA)

    assert first == repeated
    assert dart['reporting_chain'] != mala['reporting_chain']


def tuning_row(configuration, value, score):
    row = common.empty_raw_row(
        configuration, 'numpy', common.LOGISTIC_TUNING, common.DART
    )
    row.update({
        'evaluation_seed': configuration.evaluationSeeds[0],
        'dimension': configuration.dimensions[0],
        'tuning_value': value,
        'slow_direction_ess_per_iteration': score,
        'status': 'ok',
    })
    return row


def test_tuning_selects_the_best_finite_candidate():
    configuration = replace(
        benchmark.SMOKE_CONFIGURATION,
        dartGammaCandidates=(0.1, 0.2),
    )
    rows = [
        tuning_row(configuration, 0.1, 0.2),
        tuning_row(configuration, 0.2, 0.3),
    ]

    selected = benchmark.select_tuning_candidate(
        configuration,
        rows,
        common.DART,
        configuration.evaluationSeeds[0],
        configuration.dimensions[0],
    )

    assert selected['value'] == 0.2


def test_descriptive_ratio_uses_all_finite_pairs():
    rows = []
    for seed, dartValue, malaValue in ((1, 2.0, 1.0), (2, 8.0, 2.0)):
        for method, value in (
                (common.DART, dartValue), (common.MALA, malaValue)):
            row = common.empty_raw_row(
                benchmark.SMOKE_CONFIGURATION,
                'numpy', common.LOGISTIC_COMPARISON, method,
            )
            row.update({
                'evaluation_seed': seed,
                'dimension': 2,
                'status': 'ok',
                'slow_direction_ess_per_iteration': value,
            })
            rows.append(row)

    ratio, pairs, wins = benchmark.descriptive_ratio(
        rows, common.MALA, (2,)
    )

    assert ratio == pytest.approx(3.0)
    assert pairs == 2
    assert wins == 2


def test_production_cli_requires_jax():
    assert benchmark.parse_args(['--production']).backend == 'jax'
    with pytest.raises(SystemExit):
        benchmark.parse_args(['--production', '--backend', 'numpy'])


def test_smoke_benchmark_runs_all_methods():
    rows = benchmark.run_benchmark(
        benchmark.SMOKE_CONFIGURATION, get_backend('numpy')
    )

    assert len(rows) == 9
    assert {
        row['method'] for row in rows
        if row['analysis'] == common.LOGISTIC_COMPARISON
    } == set(METHODS)
