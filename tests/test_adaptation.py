import copy

import numpy as np
import pytest

from styne.backend import get_backend
from styne.mcmc import MetropolisedRandomWalk, PreconditionedCrankNicolson
from styne.mcmc.acceptance import BarkerAcceptance
from styne.mcmc.annotator import Annotator
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics, DummyDiagnostics
from styne.parameter import Vector
from styne.statistics import DiagonalCovarianceMatrix, Gaussian, IIDCovarianceMatrix
from styne.statistics.radonnikodym import RadonNikodym
from styne.statistics.interface import DensityInterface
from styne.utility import RobbinsMonroAdaptation


def random_walk(backend, variance=0.2, **settings):
    start = Vector(backend.asarray([0.4, -0.2]))
    target = Gaussian(IIDCovarianceMatrix(2, backend.asarray(1.0)),
                      Vector(backend.asarray([0.0, 0.0])))
    sampler = MetropolisedRandomWalk(
        target.density, IIDCovarianceMatrix(2, backend.asarray(variance)),
        DummyDiagnostics(), **settings,
    )
    return sampler, start


@pytest.mark.parametrize('backendName', ['numpy', 'jax', 'pytorch'])
def test_warmup_uses_the_sampler_transition_and_freezes(backendName):
    if backendName != 'numpy':
        pytest.importorskip('torch' if backendName == 'pytorch' else backendName)
    backend = get_backend(backendName)
    sampler, start = random_walk(backend, acceptance=BarkerAcceptance())
    adaptation = RobbinsMonroAdaptation(sampler, offset=0, decay=1.0)
    state = adaptation.initial_state(start)
    expected, record, _ = sampler.step(
        state.chainState, backend.random_state(7)
    )

    adapted, transition, rng = adaptation.step(state, backend.random_state(7))

    np.testing.assert_allclose(
        np.asarray(adapted.parameter.coordinate),
        np.asarray(expected.parameter.coordinate), rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(adapted.logScale),
        np.asarray(state.logScale) + float(np.asarray(record.outcome)) - 0.3,
        rtol=1e-6,
    )
    frozen = adaptation.freeze(adapted)
    assert frozen.acceptance is sampler.acceptance
    assert sampler.chain.length == frozen.chain.length == 0
    assert sampler.lastState is None
    np.testing.assert_allclose(
        np.asarray(sampler.proposalCovariance.marginalVariance), 0.2
    )
    nextState, _, _ = frozen.step(adapted.chainState, rng)
    assert nextState.parameter.dimension == 2
    assert not hasattr(nextState, 'logScale')


def test_covariance_scaling_is_part_of_the_adapted_variance():
    sampler, start = random_walk(get_backend('numpy'))
    sampler.proposalCovariance = IIDCovarianceMatrix(2, 0.2, scaling=3.0)
    adaptation = RobbinsMonroAdaptation(sampler)
    state = adaptation.initial_state(start)
    frozen = adaptation.freeze(state)

    np.testing.assert_allclose(np.exp(state.logScale), 0.6)
    np.testing.assert_allclose(frozen.proposalCovariance.to_cholesky(),
                               sampler.proposalCovariance.to_cholesky())


def test_batch_adapts_one_scale_from_independent_chain_decisions():
    sampler, _ = random_walk(get_backend('numpy'), variance=8.0)
    adaptation = RobbinsMonroAdaptation(sampler, offset=0, decay=1.0)
    state = adaptation.initial_state(Vector(np.zeros((32, 2))))
    result, transition, _ = adaptation.step(state, np.random.default_rng(3))

    assert transition.outcome.shape == (32,)
    assert np.any(transition.outcome) and not np.all(transition.outcome)
    np.testing.assert_allclose(result.logScale,
                               state.logScale + transition.outcome.mean() - 0.3)


@pytest.mark.parametrize('backendName', ['numpy', 'jax', 'pytorch'])
def test_pcn_warmup_caps_beta_and_keeps_the_reference(backendName):
    if backendName != 'numpy':
        pytest.importorskip('torch' if backendName == 'pytorch' else backendName)
    backend = get_backend(backendName)
    prior = Gaussian(IIDCovarianceMatrix(2, backend.asarray(1.0)),
                     Vector(backend.zeros(2)))

    class FlatLikelihood(DensityInterface):
        domainType = Vector
        domainDimension = 2

        def evaluate_log(self, parameter):
            return 0.0 * parameter.coordinate[..., 0]

    sampler = PreconditionedCrankNicolson(
        RadonNikodym(prior, FlatLikelihood()), 0.9, DummyDiagnostics()
    )
    adaptation = RobbinsMonroAdaptation(sampler, offset=0, decay=1)
    state = adaptation.initial_state(Vector(backend.zeros(2)))
    state, _ = adaptation.run(5, state, backend.random_state(8))

    assert state.logScale == 0.0
    assert adaptation.freeze(state).proposal.beta == 1.0
    assert adaptation.freeze(state).proposal.reference is prior
    assert sampler.proposal.beta == 0.9


@pytest.mark.parametrize('backendName', ['numpy', 'jax', 'pytorch'])
def test_freeze_hands_the_warmup_rng_to_the_production_runner(backendName):
    if backendName != 'numpy':
        pytest.importorskip('torch' if backendName == 'pytorch' else backendName)
    backend = get_backend(backendName)
    sampler, start = random_walk(backend)
    adaptation = RobbinsMonroAdaptation(sampler)
    state, rng = adaptation.run(3, adaptation.initial_state(start),
                                backend.random_state(17))
    frozen = adaptation.freeze(state, rng)
    expected, _, _ = frozen.step(state.chainState, copy.deepcopy(rng))
    frozen.run(1, state.parameter)
    np.testing.assert_allclose(np.asarray(frozen.lastState.coordinate),
                               np.asarray(expected.parameter.coordinate), atol=1e-6)


def test_jax_warmup_scan_matches_eager_steps():
    jax = pytest.importorskip('jax')
    backend = get_backend('jax')
    sampler, start = random_walk(backend)
    adaptation = RobbinsMonroAdaptation(sampler)
    state = adaptation.initial_state(start)
    scanned, scanRng = adaptation.run(8, state, jax.random.key(3))
    eager, rng = state, jax.random.key(3)
    for _ in range(8):
        eager, _, rng = adaptation.step(eager, rng)

    np.testing.assert_allclose(scanned.logScale, eager.logScale, rtol=1e-5)
    np.testing.assert_allclose(scanned.parameter.coordinate,
                               eager.parameter.coordinate, rtol=1e-5)
    np.testing.assert_array_equal(jax.random.key_data(scanRng), jax.random.key_data(rng))


def test_frozen_sampler_preserves_annotation_configuration_without_history():
    class SquaredNorm(Annotator):
        def annotate(self, parameter):
            return float(np.sum(parameter.coordinate ** 2))

    sampler, start = random_walk(get_backend('numpy'), rng=np.random.default_rng(3))
    sampler.annotator = SquaredNorm()
    sampler.run(2, start)
    originalLength = sampler.chain.length
    adaptation = RobbinsMonroAdaptation(sampler)
    frozen = adaptation.freeze(adaptation.initial_state(start))
    frozen.run(3, start)

    assert sampler.chain.length == originalLength
    assert len(frozen.chain.annotations) == frozen.chain.length == 4
    assert frozen.chain.annotations[-1] == pytest.approx(
        np.sum(frozen.lastState.coordinate ** 2)
    )


@pytest.mark.parametrize('settings', [
    {'targetAcceptance': 0}, {'targetAcceptance': 1}, {'targetAcceptance': np.nan},
    {'offset': -1}, {'offset': np.inf}, {'decay': 0.5}, {'decay': 1.1},
])
def test_invalid_schedule_is_rejected(settings):
    sampler, _ = random_walk(get_backend('numpy'))
    with pytest.raises(ValueError):
        RobbinsMonroAdaptation(sampler, **settings)


def test_unsupported_proposal_is_explicit_and_negative_runs_are_rejected():
    sampler, start = random_walk(get_backend('numpy'))
    adaptation = RobbinsMonroAdaptation(sampler)
    state = adaptation.initial_state(start)
    rng = np.random.default_rng(0)
    assert adaptation.run(0, state, rng) == (state, rng)
    with pytest.raises(ValueError):
        adaptation.run(-1, state, rng)
    sampler.proposalCovariance = DiagonalCovarianceMatrix(np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match='IID MRW or a pCN'):
        RobbinsMonroAdaptation(sampler)


def test_random_walk_warmup_reaches_the_target_acceptance():
    sampler, start = random_walk(get_backend('numpy'), variance=1e-4,
                                 rng=np.random.default_rng(4))
    adaptation = RobbinsMonroAdaptation(sampler)
    state, _ = adaptation.run(3000, adaptation.initial_state(start),
                              np.random.default_rng(1))
    frozen = adaptation.freeze(state)
    frozen.diagnostics = AcceptanceRateDiagnostics()
    frozen.run(4000, state.parameter)
    assert frozen.diagnostics.global_acceptance_rate() == pytest.approx(0.3, abs=0.05)
