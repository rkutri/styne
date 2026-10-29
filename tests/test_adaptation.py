import copy
from typing import Any, NamedTuple

import numpy as np
import pytest

from styne.backend import get_backend, infer_backend
from styne.mcmc import Adaptation, NoAdaptation
from styne.mcmc import MetropolisedRandomWalk, PreconditionedCrankNicolson
from styne.mcmc.acceptance import BarkerAcceptance
from styne.mcmc.annotator import Annotator
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics, DummyDiagnostics
from styne.mcmc.method.mala import MALAProposal, MetropolisAdjustedLangevinAlgorithm
from styne.mcmc.method.mrw import MRWFactory, MRWProposal
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


@pytest.fixture(params=['numpy', 'jax', 'pytorch'])
def backend(request):
    if request.param != 'numpy':
        pytest.importorskip('torch' if request.param == 'pytorch' else request.param)
    return get_backend(request.param)


def test_no_adaptation_preserves_transition_and_identity(backend):
    sampler, start = random_walk(backend)
    state = sampler.initial_state(start)
    expected, record, _ = sampler.step(state, backend.random_state(7))
    assert isinstance(sampler.adaptation, NoAdaptation)
    sampler.adaptation = RobbinsMonroAdaptation(offset=0, decay=1)
    initial = sampler.initial_state(start)
    adapted, transition, rng = sampler.step(initial, backend.random_state(7))
    assert type(sampler) is MetropolisedRandomWalk
    assert isinstance(sampler.proposal, MRWProposal)
    np.testing.assert_allclose(np.asarray(adapted.parameter.coordinate),
                               np.asarray(expected.parameter.coordinate), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(adapted.adaptationState.logScale),
        np.log(0.2) + float(np.asarray(record.outcome)) - 0.3, rtol=1e-6,
    )
    assert state.adaptationState is None
    assert sampler.currentState is None
    assert sampler.chain.length == 0
    np.testing.assert_allclose(
        np.asarray(sampler.proposalCovariance.marginalVariance), 0.2
    )
    frozen = sampler.freeze_adaptation(adapted, rng)
    assert type(frozen) is type(sampler)
    assert isinstance(frozen.adaptation, NoAdaptation)
    np.testing.assert_allclose(
        np.asarray(frozen.proposalCovariance.marginalVariance),
        np.exp(np.asarray(adapted.adaptationState.logScale)),
    )


def test_eager_continuation_preserves_state_and_updates_visible_proposal(backend):
    sampler, start = random_walk(backend, adaptation=RobbinsMonroAdaptation(),
                                 rng=backend.random_state(17))
    initial = sampler.initial_state(start)
    expected, rng = initial, backend.random_state(17)
    for _ in range(8):
        expected, _, rng = sampler.step(expected, rng)
    sampler.run(3, start)
    sampler.continue_run(5)
    assert sampler.currentState.adaptationState.stepCount == 8
    assert sampler.chain.length == 9
    np.testing.assert_allclose(np.asarray(sampler.lastState.coordinate),
                               np.asarray(expected.parameter.coordinate), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(sampler.proposalCovariance.marginalVariance),
        np.exp(np.asarray(expected.adaptationState.logScale)),
    )
    frozen = sampler.freeze_adaptation()
    assert frozen.chain.length == 0
    assert frozen.acceptance is sampler.acceptance
    expected, _, _ = frozen.step(frozen.initial_state(sampler.lastState),
                                 copy.deepcopy(rng))
    frozen.run(1, sampler.lastState)
    np.testing.assert_allclose(np.asarray(frozen.lastState.coordinate),
                               np.asarray(expected.parameter.coordinate), atol=1e-6)
    sampler.run(2, start)
    assert sampler.currentState.adaptationState.stepCount == 2


def test_covariance_scaling_is_part_of_adapted_variance():
    sampler, start = random_walk(get_backend('numpy'))
    sampler.proposalCovariance = IIDCovarianceMatrix(2, 0.2, scaling=3.0)
    sampler.adaptation = RobbinsMonroAdaptation()
    state = sampler.initial_state(start)
    np.testing.assert_allclose(np.exp(state.adaptationState.logScale), 0.6)
    np.testing.assert_allclose(sampler.proposal_at(state).covariance.to_cholesky(),
                               sampler.proposalCovariance.to_cholesky())


def test_batch_adapts_one_scale_from_independent_decisions():
    sampler, _ = random_walk(get_backend('numpy'), variance=8.0,
                             adaptation=RobbinsMonroAdaptation(offset=0, decay=1))
    state = sampler.initial_state(Vector(np.zeros((32, 2))))
    result, transition, _ = sampler.step(state, np.random.default_rng(3))
    assert transition.outcome.shape == (32,)
    assert np.any(transition.outcome) and not np.all(transition.outcome)
    np.testing.assert_allclose(result.adaptationState.logScale,
                               state.adaptationState.logScale
                               + transition.outcome.mean() - 0.3)


def test_pcn_caps_beta_and_keeps_reference(backend):
    prior = Gaussian(IIDCovarianceMatrix(2, backend.asarray(1.0)),
                     Vector(backend.zeros(2)))

    class FlatLikelihood(DensityInterface):
        domainType = Vector
        domainDimension = 2

        def evaluate_log(self, parameter):
            return parameter.coordinate[..., 0] * 0

    sampler = PreconditionedCrankNicolson(
        RadonNikodym(prior, FlatLikelihood()), 0.9, DummyDiagnostics(),
        adaptation=RobbinsMonroAdaptation(offset=0, decay=1),
    )
    state = sampler.initial_state(Vector(backend.zeros(2)))
    rng = backend.random_state(8)
    for _ in range(5):
        state, _, rng = sampler.step(state, rng)
    assert state.adaptationState.logScale == 0
    assert sampler.proposal_at(state).beta == 1
    assert sampler.proposal_at(state).reference is prior
    assert sampler.proposal.beta == 0.9


def test_rejection_updates_state_and_freezing_keeps_annotations():
    class ZeroAcceptance(BarkerAcceptance):
        def log_probability(self, logMHRatio):
            return np.full_like(logMHRatio, -np.inf)

    class SquaredNorm(Annotator):
        def annotate(self, parameter):
            return float(np.sum(parameter.coordinate ** 2))

    sampler, start = random_walk(get_backend('numpy'), acceptance=ZeroAcceptance(),
                                 adaptation=RobbinsMonroAdaptation(offset=0, decay=1))
    sampler.annotator = SquaredNorm()
    sampler.run(2, start)
    np.testing.assert_allclose(sampler.lastState.coordinate, start.coordinate)
    np.testing.assert_allclose(sampler.currentState.adaptationState.logScale,
                               np.log(0.2) - 0.3 - 0.15)
    fixed = sampler.freeze_adaptation()
    fixed.run(3, start)
    assert sampler.chain.length == 3
    assert len(fixed.chain.annotations) == 4
    assert fixed.chain.annotations[-1] == pytest.approx(np.sum(start.coordinate ** 2))


@pytest.mark.parametrize('settings', [
    {'targetAcceptance': 0}, {'targetAcceptance': 1}, {'targetAcceptance': np.nan},
    {'offset': -1}, {'offset': np.inf}, {'decay': 0.5}, {'decay': 1.1},
    {'stopAfter': -1}, {'stopAfter': True}, {'stopAfter': 2.5},
])
def test_invalid_schedule_is_rejected(settings):
    with pytest.raises(ValueError):
        RobbinsMonroAdaptation(**settings)


def test_strategy_validation_and_factory_configuration():
    sampler, start = random_walk(get_backend('numpy'))
    with pytest.raises(TypeError, match='Adaptation'):
        sampler.adaptation = object()
    sampler.proposalCovariance = DiagonalCovarianceMatrix(np.array([1.0, 2.0]))
    sampler.adaptation = RobbinsMonroAdaptation()
    with pytest.raises(TypeError, match='IID MRW or a pCN'):
        sampler.initial_state(start)
    factory = MRWFactory()
    factory.target = sampler.target
    factory.proposalCovariance = IIDCovarianceMatrix(2, 0.2)
    factory.adaptation = RobbinsMonroAdaptation()
    result = factory.create()
    assert type(result) is MetropolisedRandomWalk
    assert result.adaptation is factory.adaptation
    result.run(3, start)
    assert result.currentState.adaptationState.stepCount == 3
    result.adaptation = NoAdaptation()
    assert result.currentState is None
    assert result.chain.length == 0


def test_robbins_monro_reaches_target_acceptance():
    sampler, start = random_walk(get_backend('numpy'), variance=1e-4,
                                 adaptation=RobbinsMonroAdaptation(),
                                 rng=np.random.default_rng(1))
    sampler.storeChain = False
    sampler.run(3000, start)
    frozen = sampler.freeze_adaptation(rng=np.random.default_rng(4))
    frozen.diagnostics = AcceptanceRateDiagnostics()
    frozen.run(4000, sampler.lastState)
    assert frozen.diagnostics.global_acceptance_rate() == pytest.approx(0.3, abs=0.05)


class MALAScaleState(NamedTuple):
    stepSize: Any
    count: Any


class MALAScaleAdaptation(Adaptation):
    def initial_state(self, parameter, proposalMethod):
        backend = infer_backend(parameter.coordinate)
        return MALAScaleState(backend.asarray(proposalMethod.stepSize),
                               backend.asarray(0))

    def prepare_proposal(self, proposalMethod, state):
        return MALAProposal(
            2, state.stepSize, logGradientCallable=lambda x: -x.coordinate
        )

    def update_state(self, state, nextState, transitionData):
        return MALAScaleState(state.stepSize * 0.9, state.count + 1)


def test_compiled_mala_uses_dynamic_proposal_settings():
    jax = pytest.importorskip('jax')
    backend = get_backend('jax')
    reference, start = random_walk(backend)
    sampler = MetropolisAdjustedLangevinAlgorithm(
        reference.target, 0.5, DummyDiagnostics(), adaptation=MALAScaleAdaptation(),
    )
    initial = sampler.initial_state(start)
    compiledStep = jax.jit(sampler.step)
    from dataclasses import replace
    for scale in (0.1, 0.8):
        state = replace(initial, adaptationState=MALAScaleState(
            backend.asarray(scale), initial.adaptationState.count))
        actual, record, _ = compiledStep(state, backend.random_state(6))
        expected, expectedRecord, _ = sampler.step(state, backend.random_state(6))
        np.testing.assert_allclose(actual.parameter.coordinate,
                                   expected.parameter.coordinate, atol=1e-6)
        np.testing.assert_allclose(record.proposed.parameter.coordinate,
                                   expectedRecord.proposed.parameter.coordinate,
                                   atol=1e-6)
    execute = sampler.compiled_trajectory(4)
    first = execute(start, backend.random_state(13))
    second = execute(first.finalState, first.rng)
    assert type(sampler) is MetropolisAdjustedLangevinAlgorithm
    assert second.finalState.adaptationState.count == 8
    np.testing.assert_allclose(second.finalState.adaptationState.stepSize,
                               0.5 * 0.9 ** 8, rtol=1e-5)


@pytest.mark.parametrize('stopAfter', [None, 0, 5])
def test_compiled_continuation_matches_eager_and_respects_stopping(stopAfter):
    jax = pytest.importorskip('jax')
    backend = get_backend('jax')
    sampler, start = random_walk(backend, adaptation=RobbinsMonroAdaptation(
        stopAfter=stopAfter))
    execute = sampler.compiled_trajectory(4)
    key = backend.random_state(13)
    first = execute(start, key)
    second = execute(first.finalState, first.rng)
    eager, rng = sampler.initial_state(start), key
    for _ in range(8):
        eager, _, rng = sampler.step(eager, rng)
    for actual, expected in zip(jax.tree_util.tree_leaves(second.finalState),
                                jax.tree_util.tree_leaves(eager)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(jax.random.key_data(second.rng),
                                   jax.random.key_data(rng))
    assert second.finalState.adaptationState.stepCount == (
        8 if stopAfter is None else stopAfter)
    assert sampler.currentState is None
    assert sampler.chain.length == 0
