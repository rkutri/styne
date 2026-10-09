from typing import Optional

from numpy.random import Generator

from styne.backend import infer_backend
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.diagnostics import ChainDiagnostics
from styne.mcmc.acceptance import AcceptanceProbability
from styne.parameter.parameter import Parameter
from styne.statistics.gaussian import Gaussian
from styne.statistics.interface import DensityInterface
from styne.statistics.covariance import DenseCovarianceMatrix


class DirectDARTProposal(ProposalMethod):
    """Gaussian DART proposal based on a fixed surrogate.

    Its correction relative to the Lebesgue measure combines the tempered
    surrogate ratio with the closed-form normalising ratio.
    """

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
        z = proposal.coordinate
        bz = self._tempering * self._precisionMean + self._gamma * z
        muz = self._proposalMeasure.covariance.apply(bz)
        namespace = infer_backend(x).namespace
        xHat = self._surrogate.mean.coordinate
        surrogateCovariance = self._surrogate.covariance
        quadSurrogateDiff = 0.5 * self._tempering * (
            surrogateCovariance.dual_quadratic_form(z - xHat)
            - surrogateCovariance.dual_quadratic_form(x - xHat)
        )
        normDiff = 0.5 * (
            namespace.sum(mux * bx, axis=-1)
            - namespace.sum(muz * bz, axis=-1)
        ) - 0.5 * self._gamma * (
            namespace.sum(x * x, axis=-1)
            - namespace.sum(z * z, axis=-1)
        )
        return self.record(
            state, proposal, quadSurrogateDiff + normDiff
        ), nextRng


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
