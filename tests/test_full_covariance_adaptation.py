import numpy as np
import pytest

from styne.backend import get_backend
from styne.mcmc import MetropolisedRandomWalk
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.method.mrw import MRWProposal
from styne.parameter import Vector
from styne.statistics import (
    DenseCovarianceMatrix, DiagonalCovarianceMatrix, Gaussian, IIDCovarianceMatrix,
)
from styne.utility import FullCovarianceAdaptation


@pytest.fixture(params=['numpy', 'jax', 'pytorch'])
def backend(request):
    if request.param != 'numpy':
        pytest.importorskip('torch' if request.param == 'pytorch' else request.param)
    return get_backend(request.param)


def random_walk(backend, interval=10, nugget=0.02, **settings):
    start = Vector(backend.asarray([0.2, -0.3], dtype='float32'))
    target = Gaussian(
        DiagonalCovarianceMatrix(backend.asarray([1.0, 2.0], dtype='float32')),
        Vector(backend.zeros(2, dtype='float32')),
    )
    sampler = MetropolisedRandomWalk(
        target.density,
        DiagonalCovarianceMatrix(backend.asarray([0.1, 0.2], dtype='float32'),
                                 scaling=3.0),
        DummyDiagnostics(), rng=backend.random_state(11),
        adaptation=FullCovarianceAdaptation(interval=interval, nugget=nugget),
        **settings,
    )
    return sampler, start


@pytest.mark.parametrize('interval', [1, 3, 10])
def test_factor_changes_only_on_schedule_and_counts_retained_positions(
        backend, interval):
    sampler, start = random_walk(backend, interval)
    state = sampler.initial_state(start)
    initial = state
    previousFactor = np.asarray(state.adaptationState.cholFactor).copy()
    np.testing.assert_allclose(previousFactor @ previousFactor.T,
                               np.diag([0.3, 0.6]), atol=1e-6)
    metadata = backend.metadata(state.adaptationState.cholFactor)
    retained = []
    rng = backend.random_state(11)
    for step in range(1, 16):
        previous = state
        previousM2 = np.asarray(previous.adaptationState.moments.m2).copy()
        state, _, rng = sampler.step(state, rng)
        retained.append(np.asarray(state.parameter.coordinate))
        adaptation = state.adaptationState
        factor = np.asarray(adaptation.cholFactor)
        assert adaptation.moments.nData == step
        assert previous.adaptationState.moments.nData == step - 1
        np.testing.assert_array_equal(previous.adaptationState.moments.m2, previousM2)
        if step >= 2 and step % interval == 0:
            expected = np.cov(retained, rowvar=False) + 0.02 * np.eye(2)
            np.testing.assert_allclose(factor @ factor.T, expected, atol=1e-6)
        else:
            np.testing.assert_array_equal(factor, previousFactor)
        assert backend.metadata(adaptation.cholFactor) == metadata
        assert isinstance(sampler.proposal_at(state).covariance, DenseCovarianceMatrix)
        previousFactor = factor.copy()
    assert initial.adaptationState.moments.nData == 0
    np.testing.assert_array_equal(initial.adaptationState.moments.m2, 0)


def test_eager_factorisation_count_includes_continuation_but_not_preparation_or_freezing(
        monkeypatch):
    backend = get_backend('numpy')
    sampler, start = random_walk(backend)
    original = backend.cholesky
    calls = []

    def count_cholesky(matrix):
        calls.append(matrix.copy())
        return original(matrix)

    monkeypatch.setattr(backend, 'cholesky', count_cholesky)
    sampler.run(25, start)
    assert len(calls) == 2
    sampler.proposal_at(sampler.currentState)
    sampler.proposalCovariance
    fixed = sampler.freeze_adaptation()
    fixed.run(5, sampler.lastState)
    assert len(calls) == 2
    sampler.continue_run(5)
    assert len(calls) == 3
    assert sampler.currentState.adaptationState.moments.nData == 30
    retained = np.asarray(sampler.chain.trajectory)[1:]
    np.testing.assert_allclose(calls[-1], np.cov(retained, rowvar=False)
                               + 0.02 * np.eye(2), atol=1e-6)


def test_compiled_jax_executes_only_scheduled_factorisations(monkeypatch):
    jax = pytest.importorskip('jax')
    backend = get_backend('jax')
    sampler, start = random_walk(backend, interval=5)
    original = backend.cholesky
    calls = []

    def count_cholesky(matrix):
        # Count execution, because Python counters would count branch tracing.
        jax.debug.callback(lambda value: calls.append(np.asarray(value)), matrix,
                           ordered=True)
        return original(matrix)

    monkeypatch.setattr(backend, 'cholesky', count_cholesky)
    execute = sampler.compiled_trajectory(6)
    first = execute(start, backend.random_state(11))
    jax.block_until_ready(first.finalState)
    jax.effects_barrier()
    assert len(calls) == 1
    second = execute(first.finalState, first.rng)
    jax.block_until_ready(second.finalState)
    jax.effects_barrier()
    assert len(calls) == 2
    sampler.proposal_at(second.finalState)
    fixed = sampler.freeze_adaptation(second.finalState, second.rng)
    fixedResult = fixed.compiled_trajectory(6)(second.final, second.rng)
    jax.block_until_ready(fixedResult.finalState)
    jax.effects_barrier()
    assert len(calls) == 2
    third = execute(second.finalState, second.rng)
    jax.block_until_ready(third.finalState)
    jax.effects_barrier()
    assert len(calls) == 3
    assert third.finalState.adaptationState.moments.nData == 18
    retained = np.concatenate([np.asarray(result.coordinates)
                               for result in (first, second, third)])
    expected = np.cov(retained[:15], rowvar=False) + 0.02 * np.eye(2)
    factor = np.asarray(third.finalState.adaptationState.cholFactor)
    np.testing.assert_allclose(factor @ factor.T, expected, atol=1e-6)


@pytest.mark.parametrize('nugget', [0.02, 0.1])
def test_rejections_are_repeated_observations(backend, nugget):
    class AlwaysReject(AcceptanceProbability):
        def log_probability(self, logMHRatio):
            metadata = backend.metadata(logMHRatio)
            return backend.full(logMHRatio.shape, -np.inf, dtype=metadata.dtype,
                                device=metadata.device)

    sampler, start = random_walk(backend, interval=2, nugget=nugget,
                                 acceptance=AlwaysReject())
    sampler.run(6, start)
    record = sampler.currentState.adaptationState
    assert record.moments.nData == 6
    np.testing.assert_array_equal(record.moments.mean, start.coordinate)
    np.testing.assert_array_equal(record.moments.m2, 0)
    np.testing.assert_allclose(record.cholFactor, np.sqrt(nugget) * np.eye(2), atol=1e-6)


@pytest.mark.parametrize('settings', [
    {'interval': 0}, {'interval': -1}, {'interval': True}, {'interval': 2.5},
    {'nugget': 0}, {'nugget': -1}, {'nugget': np.inf}, {'nugget': np.nan},
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        FullCovarianceAdaptation(**settings)


def test_non_mrw_proposals_and_batched_observations_are_rejected():
    backend = get_backend('numpy')
    strategy = FullCovarianceAdaptation()
    with pytest.raises(TypeError, match='MRW proposal'):
        strategy.initial_state(Vector(np.zeros(2)), object())
    with pytest.raises(ValueError, match='one-dimensional observation'):
        strategy.initial_state(Vector(backend.zeros((3, 2))),
                               MRWProposal(IIDCovarianceMatrix(2, 0.1)))


def test_compiled_trajectory_reports_failed_adaptation_at_call_boundary():
    pytest.importorskip('jax')
    from dataclasses import replace
    from styne.statistics import FullCovarianceAccumulator
    from styne.utility import NotPositiveDefinite
    backend = get_backend('jax')

    class AlwaysReject(AcceptanceProbability):
        def log_probability(self, logMHRatio):
            return backend.full(logMHRatio.shape, -np.inf, dtype='float32')

    sampler, _ = random_walk(backend, interval=2, nugget=0.01,
                              acceptance=AlwaysReject())
    start = Vector(backend.full(2, 1000.0, dtype='float32'))
    state = sampler.initial_state(start)
    moments = FullCovarianceAccumulator.update_record(
        state.adaptationState.moments, backend.zeros(2, dtype='float32'),
    )
    state = replace(
        state, adaptationState=state.adaptationState._replace(moments=moments),
    )
    execute = sampler.compiled_trajectory(2)
    with pytest.raises(NotPositiveDefinite, match='nugget=0.01'):
        execute(state, backend.random_state(11))
    assert state.adaptationState.moments.nData == 1
