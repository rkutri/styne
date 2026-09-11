import numpy as np
import pytest

from benchmarks import dartbenchmark as common
from benchmarks import dartlocalisation as benchmark
from styne.backend import get_backend
from styne.mcmc.transition import TransitionData
from styne.parameter import Vector


def test_production_protocol_is_fixed():
    configuration = benchmark.PRODUCTION_CONFIGURATION

    assert configuration.ablationDimensions == (4, 6, 8, 10, 12, 14, 16)
    assert configuration.ablationEvaluationSeeds == tuple(
        range(43_000, 45_000, 100)
    )
    assert configuration.ablationPilotSeeds == (
        23_000, 23_100, 23_200, 23_300
    )
    assert configuration.ablationRetained == 30_000

    pilotRows = (
        len(common.GAMMA_CANDIDATES)
        * len(configuration.ablationPilotSeeds)
        * len(configuration.ablationDimensions)
        * len(configuration.ablationConditions)
    )
    reportingRows = (
        2
        * len(configuration.ablationEvaluationSeeds)
        * len(configuration.ablationDimensions)
        * len(configuration.ablationConditions)
    )
    assert pilotRows == 560
    assert reportingRows == 1_120


def test_unlocalised_ratio_matches_independence_mh():
    backend = get_backend('numpy')
    problem = common.generate_gaussian_problem(41_004, 4)
    target, surrogate, precision = common.gaussian_components(
        backend, problem, 'combined_misspecification', 42_004
    )
    sampler = common.UnlocalisedGaussianSurrogate(
        target, common.TEMPERING, surrogate
    )
    state = Vector(np.array([0.2, -0.1, 0.4, 0.3]))
    proposal = Vector(np.array([-0.3, 0.5, 0.1, -0.2]))
    transition = TransitionData(
        current=sampler.evaluate_state(state),
        proposed=sampler.evaluate_state(proposal),
    )
    mean = np.asarray(surrogate.mean.coordinate)
    stateDifference = state.coordinate - mean
    proposalDifference = proposal.coordinate - mean
    expected = (
        target.evaluate_log(proposal)
        - target.evaluate_log(state)
        + 0.5 * common.TEMPERING * (
            proposalDifference @ precision @ proposalDifference
            - stateDifference @ precision @ stateDifference
        )
    )

    assert sampler._log_mh_ratio(transition) == pytest.approx(expected)


def test_configuration_ratios_use_ratios_of_means():
    rows = []
    for seed, localised, unlocalised in ((1, 4.0, 2.0), (2, 8.0, 2.0)):
        for method, value in (
                (common.LOCALISED, localised),
                (common.UNLOCALISED, unlocalised)):
            row = common.empty_raw_row(
                benchmark.SMOKE_CONFIGURATION,
                'numpy', common.LOCALISATION_ABLATION, method,
            )
            row.update({
                'evaluation_seed': seed,
                'dimension': 2,
                'surrogate_condition': 'laplace',
                'status': 'ok',
                common.PRIMARY_ABLATION_METRIC: value,
            })
            rows.append(row)

    ratios = benchmark.configuration_ratios(rows)

    assert ratios[(2, 'laplace')] == pytest.approx(3.0)


def test_production_cli_requires_jax():
    assert benchmark.parse_args(['--production']).backend == 'jax'
    with pytest.raises(SystemExit):
        benchmark.parse_args(['--production', '--backend', 'numpy'])


def test_smoke_benchmark_runs_both_variants():
    rows, selectedGamma, _ = benchmark.run_benchmark(
        benchmark.SMOKE_CONFIGURATION, get_backend('numpy')
    )

    assert len(rows) == 28
    assert selectedGamma in common.GAMMA_CANDIDATES
    assert {
        row['method'] for row in rows
        if row['analysis'] == common.LOCALISATION_ABLATION
    } == {common.LOCALISED, common.UNLOCALISED}
