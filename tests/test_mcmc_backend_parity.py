import numpy as np
import pytest

from styne.backend import BackendUnavailableError, get_backend, infer_backend
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.transition import TransitionData
from styne.parameter import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import GaussianDensity


@pytest.fixture(params=('numpy', 'pytorch', 'jax'))
def backend(request):
    try:
        return get_backend(request.param)
    except BackendUnavailableError:
        pytest.skip(f'{request.param} is not installed')


class CountingDensity:

    def __init__(self, density):
        self._density = density
        self.evaluations = 0

    def evaluate_log(self, parameter):
        self.evaluations += 1
        return self._density.evaluate_log(parameter)


class FixedOffsetProposal(ProposalMethod):

    def propose(self, state, rng):
        proposal = state.with_coordinate(state.coordinate + 1.0)
        return TransitionData(state, proposal), rng


class SymmetricMetropolisHastings(MetropolisHastings):

    def _log_mh_ratio(self, transition):
        return transition.proposed.logDensity - transition.current.logDensity


def make_sampler(backend):
    variance = backend.asarray(1.0, dtype='float32')
    target = CountingDensity(GaussianDensity(
        IIDCovarianceMatrix(1, variance),
        Vector(backend.zeros(1, dtype='float32')),
    ))
    sampler = SymmetricMetropolisHastings(
        target, FixedOffsetProposal(), DummyDiagnostics(),
    )
    return sampler, target


@pytest.mark.parametrize(
    ('uniform', 'accepted'),
    ((0.5, True), (0.9, False)),
)
def test_mh_transition_has_deterministic_cross_backend_parity(
        backend, monkeypatch, uniform, accepted):
    def fixed_uniform(randomState, shape, *, dtype=None, device=None):
        return backend.full(shape, uniform, dtype=dtype, device=device), randomState

    monkeypatch.setattr(backend, 'uniform', fixed_uniform)
    sampler, target = make_sampler(backend)
    current = sampler.initial_state(Vector(backend.zeros(1, dtype='float32')))

    nextState, transition, rng = sampler.step(
        current, backend.random_state(7)
    )
    repeatedState, repeatedTransition, _ = sampler.step(nextState, rng)

    assert infer_backend(nextState.parameter.coordinate) is backend
    assert transition.outcome.shape == ()
    assert bool(transition.outcome) is accepted
    assert not bool(repeatedTransition.outcome)
    assert target.evaluations == 3
    np.testing.assert_allclose(transition.logAcceptanceProbability, -0.5)
    np.testing.assert_allclose(
        nextState.parameter.coordinate,
        [1.0] if accepted else [0.0],
    )
    np.testing.assert_allclose(
        repeatedState.parameter.coordinate,
        [1.0] if accepted else [0.0],
    )


def test_mh_transition_preserves_batched_backend_state(backend, monkeypatch):
    def fixed_uniform(randomState, shape, *, dtype=None, device=None):
        return backend.full(shape, 0.5, dtype=dtype, device=device), randomState

    monkeypatch.setattr(backend, 'uniform', fixed_uniform)
    sampler, _ = make_sampler(backend)
    current = sampler.initial_state(
        Vector(backend.zeros((2, 1), dtype='float32'))
    )

    nextState, transition, _ = sampler.step(current, backend.random_state(7))

    assert nextState.parameter.coordinate.shape == (2, 1)
    assert transition.outcome.shape == (2,)
    np.testing.assert_allclose(nextState.parameter.coordinate, [[1.0], [1.0]])
