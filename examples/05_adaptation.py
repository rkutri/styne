"""Learn a diagonal MRW proposal covariance from the chain's previous states.

Run: python examples/05_adaptation.py --backend numpy --smoke
"""
from typing import Any, NamedTuple

from styne.backend import infer_backend
from styne.mcmc import Adaptation, MetropolisedRandomWalk
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.mcmc.method.mrw import MRWProposal
from styne.parameter import Vector
from styne.statistics import Gaussian, IIDCovarianceMatrix, DiagonalCovarianceMatrix
from styne.statistics import WelfordAccumulator
from styne.utility import RobbinsMonroAdaptation


# Keep changing quantities in an immutable state, not on the strategy object.
# Fixed array shapes and dtypes let the same rule run inside JAX compilation.
class CovarianceAdaptationState(NamedTuple):
    marginalVariance: Any
    moments: Any


# A custom strategy implements these three methods; MH decides when to call them.
class CovarianceAdaptation(Adaptation):
    """Single-chain, diagonal MRW example; interval and nugget must be positive."""

    def __init__(self, interval=10, nugget=0.01):
        self._interval = interval
        self._nugget = nugget

    def initial_state(self, parameter, proposalMethod):
        # Start with the supplied diagonal covariance and empty empirical moments.
        covariance = proposalMethod.covariance
        backend = infer_backend(parameter.coordinate)
        metadata = backend.metadata(parameter.coordinate)
        variance = backend.asarray(covariance.scaling * covariance.marginalVariance,
                                   dtype=metadata.dtype, device=metadata.device)
        return CovarianceAdaptationState(
            variance, WelfordAccumulator.initial_record(parameter.coordinate),
        )

    def prepare_proposal(self, proposalMethod, state):
        # MH uses this proposal for the whole transition, including its correction.
        return MRWProposal(DiagonalCovarianceMatrix(state.marginalVariance))

    def update_state(self, state, nextState, transitionData):
        # MH supplies the retained state, so rejections count as repeated samples.
        moments = WelfordAccumulator.update_record(
            state.moments, nextState.parameter.coordinate
        )
        backend = infer_backend(state.marginalVariance)
        metadata = backend.metadata(state.marginalVariance)
        namespace = backend.namespace
        count = backend.asarray(moments.nData, dtype=metadata.dtype,
                                device=metadata.device)
        # The nugget keeps variances positive; array selection also works under JAX.
        variance = moments.m2 / namespace.maximum(count - 1, 1) + self._nugget
        due = (moments.nData >= 2) & (moments.nData % self._interval == 0)
        variance = namespace.where(due, variance, state.marginalVariance)
        return CovarianceAdaptationState(variance, moments)


if __name__ == '__main__':
    from runtime import configure_backend, parse_arguments

    arguments = parse_arguments(__doc__)
    backend, rng = configure_backend(arguments.backend)
    start = Vector(backend.zeros(2, dtype='float32'))
    target = Gaussian(DiagonalCovarianceMatrix(backend.asarray([1.0, 4.0])), start)
    # Choose our rule, or an existing strategy that adapts one scalar scale.
    adaptation = CovarianceAdaptation(interval=10, nugget=0.01)
    # adaptation = RobbinsMonroAdaptation(targetAcceptance=0.3)
    sampler = MetropolisedRandomWalk(
        target.density, IIDCovarianceMatrix(2, backend.asarray(0.1)),
        AcceptanceRateDiagnostics(), rng=rng, adaptation=adaptation,
    )
    nSteps = 40 if arguments.smoke else 1000

    # The sampler remains an ordinary MRW. MH handles the three methods above.
    sampler.run(nSteps, start)
    sampler.continue_run(nSteps)  # Keep learning, retaining the adaptation history.

    # Freezing is optional: create a fixed-proposal sampler for subsequent draws.
    fixed = sampler.freeze_adaptation()
    fixed.run(nSteps, sampler.lastState)
    print(f'Learned proposal variances: {sampler.proposalCovariance.marginalVariance}')
