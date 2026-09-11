import numpy as np

from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.transition import TransitionData
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import GaussianDensity


class FixedUniformState:
    def __init__(self, value):
        self._value = value

    def uniform(self, size):
        return np.full(size, self._value)


class FixedOffsetProposal(ProposalMethod):
    def propose(self, state, rng):
        return TransitionData(state, state.with_coordinate(
            state.coordinate + 1.0
        )), rng


class SymmetricMetropolisHastings(MetropolisHastings):
    def _log_mh_ratio(self, transition):
        return transition.proposed.logDensity - transition.current.logDensity


def make_sampler():
    target = GaussianDensity(IIDCovarianceMatrix(1, 1.0), Vector([0.0]))
    return SymmetricMetropolisHastings(
        target, FixedOffsetProposal(), DummyDiagnostics()
    )


def test_deterministic_mh_step_accepts_with_fixed_uniform():
    sampler = make_sampler()
    current = sampler.initial_state(Vector([0.0]))

    nextState, transition, _ = sampler.step(current, FixedUniformState(0.1))

    assert transition.outcome
    np.testing.assert_array_equal(nextState.parameter.coordinate, [1.0])


def test_deterministic_mh_step_rejects_with_fixed_uniform():
    sampler = make_sampler()
    current = sampler.initial_state(Vector([0.0]))

    nextState, transition, _ = sampler.step(current, FixedUniformState(0.9))

    assert not transition.outcome
    np.testing.assert_array_equal(nextState.parameter.coordinate, [0.0])
