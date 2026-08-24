from typing import Optional

from numpy.random import Generator

from styne.backend import infer_backend
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.transition import TransitionData
from styne.mcmc.diagnostics import ChainDiagnostics
from styne.mcmc.acceptance import AcceptanceProbability
from styne.parameter.parameter import Parameter
from styne.statistics.gaussian import Gaussian
from styne.statistics.interface import DensityInterface
from styne.statistics.covariance import DenseCovarianceMatrix


class DirectDARTProposal(ProposalMethod):
    """Gaussian DART proposal based on a fixed surrogate."""

    def __init__(
            self, tempering: float, gamma: float, surrogate: Gaussian,
            proposalCovariance: DenseCovarianceMatrix):
        self._tempering = tempering
        self._gamma = gamma
        self._surrogate = surrogate
        self._precisionMean = self._surrogate.covariance.apply_inverse(
            self._surrogate.mean.coordinate
        )
        self._proposalMeasure = Gaussian(proposalCovariance)

    def propose(self, state: Parameter, rng):
        x = state.coordinate
        bx = self._tempering * self._precisionMean + self._gamma * x
        mux = self._proposalMeasure.covariance.apply(bx)
        propMean = state.with_coordinate(mux)
        proposal, nextRng = self._proposalMeasure.with_mean(propMean).sample(
            rng
        )
        return (
            TransitionData(state, proposal, auxiliary={'bx': bx, 'mux': mux}),
            nextRng,
        )


class DirectDART(MetropolisHastings):
    """Data-assimilation based regularised transition sampler."""

    def __init__(
            self, targetDensity: DensityInterface, tempering: float,
            gamma: float, surrogate: Gaussian,
            proposalCovariance: DenseCovarianceMatrix,
            diagnostics: ChainDiagnostics,
            acceptance: AcceptanceProbability = None,
            rng: Optional[Generator] = None):
        if tempering <= 0.0 or tempering > 1.0:
            raise ValueError('Tempering parameter must be in (0, 1].')
        if gamma <= 0.0:
            raise ValueError('Regularisation gamma must be strictly positive.')

        self._tempering = tempering
        self._gamma = gamma
        self._surrogate = surrogate

        proposalMethod = DirectDARTProposal(
            tempering, gamma, surrogate, proposalCovariance
        )
        super().__init__(
            targetDensity,
            proposalMethod,
            diagnostics,
            acceptance=acceptance,
            rng=rng,
        )

    def _log_mh_ratio(self, transition: TransitionData):
        x = transition.state.coordinate
        z = transition.proposal.coordinate
        backend = infer_backend(x)
        namespace = backend.namespace
        metadata = backend.metadata(x)
        negativeInfinity = backend.asarray(
            float('-inf'), dtype=metadata.dtype, device=metadata.device
        )

        logTargetDiff = (
            transition.proposed.logDensity - transition.current.logDensity
        )

        xHat = self._surrogate.mean.coordinate

        diffZ = z - xHat
        diffX = x - xHat
        quadDiffZ = self._surrogate.covariance.dual_quadratic_form(diffZ)
        quadDiffX = self._surrogate.covariance.dual_quadratic_form(diffX)
        quadSurrogateDiff = 0.5 * self._tempering * (quadDiffZ - quadDiffX)

        bx = transition.auxiliary['bx']
        mux = transition.auxiliary['mux']
        bz = (
            self._tempering * self._proposalMethod._precisionMean
            + self._gamma * z
        )
        muz = self._proposalMethod._proposalMeasure.covariance.apply(bz)

        normDiff = 0.5 * (
            namespace.sum(mux * bx, axis=-1)
            - namespace.sum(muz * bz, axis=-1)
        ) - 0.5 * self._gamma * (
            namespace.sum(x * x, axis=-1)
            - namespace.sum(z * z, axis=-1)
        )

        logRatio = logTargetDiff + quadSurrogateDiff + normDiff
        return namespace.where(
            logTargetDiff == negativeInfinity, negativeInfinity, logRatio
        )
