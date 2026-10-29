"""Adapt a diagonal MRW proposal using empirical marginal variances.

Run: python examples/adaptation.py --backend jax --smoke
See docs/mcmc.md for the strategy contract and its numerical state.
"""
import argparse
from typing import Any, NamedTuple

from styne.backend import get_backend, infer_backend
from styne.mcmc import Adaptation, MetropolisedRandomWalk
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.mcmc.method.mrw import MRWProposal
from styne.parameter import Vector
from styne.statistics import Gaussian, IIDCovarianceMatrix, DiagonalCovarianceMatrix
from styne.statistics import WelfordAccumulator


class CovarianceAdaptationState(NamedTuple):
    """Proposal variances and moments of all retained chain states."""

    marginalVariance: Any
    moments: Any


class CovarianceAdaptation(Adaptation):
    """Update a single MRW chain's diagonal covariance every 'interval' steps.

    A positive nugget keeps the proposal nonsingular. This illustrates
    the interface; it is not a convergence guarantee or a tuned default method.
    """

    def __init__(self, interval=10, nugget=0.01):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("interval must be a positive integer.")
        if not 0 < nugget < float('inf'):
            raise ValueError("nugget must be finite and positive.")
        self._interval = interval
        self._nugget = nugget

    def initial_state(self, parameter, proposalMethod):
        if not isinstance(proposalMethod, MRWProposal):
            raise TypeError("CovarianceAdaptation requires an MRW proposal.")
        covariance = proposalMethod.covariance
        if not isinstance(covariance, DiagonalCovarianceMatrix):
            raise TypeError("This example requires an initial diagonal covariance.")
        if parameter.coordinate.ndim != 1:
            raise ValueError("This example adapts one chain, without batch axes.")
        backend = infer_backend(parameter.coordinate)
        metadata = backend.metadata(parameter.coordinate)
        variance = backend.asarray(covariance.scaling * covariance.marginalVariance,
                                   dtype=metadata.dtype, device=metadata.device)
        return CovarianceAdaptationState(
            variance, WelfordAccumulator.initial_record(parameter.coordinate),
        )

    def prepare_proposal(self, proposalMethod, state):
        return MRWProposal(DiagonalCovarianceMatrix(state.marginalVariance))

    def update_state(self, state, nextState, transitionData):
        moments = WelfordAccumulator.update_record(
            state.moments, nextState.parameter.coordinate
        )
        backend = infer_backend(state.marginalVariance)
        metadata = backend.metadata(state.marginalVariance)
        namespace = backend.namespace
        count = backend.asarray(moments.nData, dtype=metadata.dtype,
                                device=metadata.device)
        variance = moments.m2 / namespace.maximum(count - 1, 1) + self._nugget
        due = (moments.nData >= 2) & (moments.nData % self._interval == 0)
        variance = namespace.where(due, variance, state.marginalVariance)
        return CovarianceAdaptationState(variance, moments)


def run(backendName, smoke=False):
    backend = get_backend(backendName)
    start = Vector(backend.zeros(2, dtype='float32'))
    target = Gaussian(DiagonalCovarianceMatrix(backend.asarray([1.0, 4.0])), start)
    sampler = MetropolisedRandomWalk(
        target.density, IIDCovarianceMatrix(2, backend.asarray(0.1)),
        AcceptanceRateDiagnostics(), rng=backend.random_state(7),
        adaptation=CovarianceAdaptation(interval=10),
    )
    nSteps = 40 if smoke else 2000
    if backend.capabilities.transformedLoops:
        execute = sampler.compiled_trajectory(nSteps)
        first = execute(start, backend.random_state(11))
        second = execute(first.finalState, first.rng)
        state, rng = second.finalState, second.rng
        fixed = sampler.freeze_adaptation(state, rng)
    else:
        sampler.run(nSteps, start)
        sampler.continue_run(nSteps)
        state = sampler.currentState
        fixed = sampler.freeze_adaptation()
    assert isinstance(sampler, MetropolisedRandomWalk)
    assert isinstance(sampler.proposal_at(state), MRWProposal)
    assert state.adaptationState.moments.nData == 2 * nSteps
    # Continuing the same sampler would keep adapting. A fixed copy is optional.
    fixed.run(nSteps, state.parameter)
    return fixed.diagnostics.global_acceptance_rate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'jax', 'pytorch'], default='numpy')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    acceptance = run(args.backend, args.smoke)
    print(f'{args.backend}: fixed-proposal acceptance {acceptance:.3f}')
