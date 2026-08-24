import numpy as np
from typing import Optional
from numpy.random import Generator

from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.factory import MHFactory
from styne.mcmc.transition import TransitionData
from styne.parameter.parameter import Parameter
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import Gaussian


class MALAProposal(ProposalMethod):
    """
    Langevin proposal for MALA.

    Given step size h (standard deviation), the proposal from state x is

        z = x + (h^2 / 2) * grad log pi(x) + h * xi,   xi ~ N(0, I)

    Parameters
    ----------
    dim : int
        Dimension of the state space.
    stepSize : float
        Step size h, interpreted as the standard deviation of the
        isotropic noise.
    logGradientCallable : callable
        Maps a Parameter to the gradient of the log target density
        (as an ndarray).
    """

    def __init__(self, dim, stepSize, logGradientCallable):

        super().__init__()

        if not callable(logGradientCallable):
            raise ValueError("gradient must be callable")

        self._h = float(stepSize)
        self._h2 = self._h * self._h
        self._logGradient = logGradientCallable

        propCov = IIDCovarianceMatrix(dim, self._h2)
        self._proposalMeasure = Gaussian(propCov)

    @property
    def stepSize(self) -> float:
        return self._h

    def _drift(self, state: Parameter) -> np.ndarray:
        """Compute the deterministic drift: x + (h^2 / 2) * grad log pi(x)."""
        return state.coordinate + 0.5 * self._h2 * self._logGradient(state)

    def generate_proposal(self, rng: Generator) -> TransitionData:
        # Guard against use outside the MH loop, where state may not be set.
        if self._state is None:
            raise ValueError(
                "Trying to generate proposal with undefined state"
            )

        driftVector = self._drift(self._state)
        propMean = self._state.with_coordinate(
            np.asarray(driftVector, dtype=np.float64)
        )

        proposal = self._proposalMeasure.with_mean(propMean).generate_realisation(rng=rng)
        return TransitionData(
            self._state, proposal, auxiliary={'drift': driftVector}
        )


class MetropolisAdjustedLangevinAlgorithm(MetropolisHastings):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA).

    Uses a Langevin-diffusion-inspired proposal with a Metropolis-Hastings
    correction to ensure the correct stationary distribution.

    Parameters
    ----------
    targetDensity : DensityInterface
        Target density. Must provide an ``evaluate_log_gradient(state)``
        method returning the gradient of the log-density as an ndarray.
        This is checked at construction time via duck typing, no specific
        base class is required, any object with the method will work.
    stepSize : float
        Step size h (standard deviation of the isotropic noise).
    diagnostics : ChainDiagnostics
        Tracks transition statistics.
    """
    name = "MALA"

    def __init__(self, targetDensity, stepSize, diagnostics,
                 acceptance: AcceptanceProbability = None,
                 rng: Optional[Generator] = None):

        if not callable(getattr(targetDensity, "evaluate_log_gradient", None)):
            raise ValueError(
                "MALA requires a target density with evaluate_log_gradient."
            )

        proposalMethod = MALAProposal(
            targetDensity.domainDimension,
            stepSize,
            targetDensity.evaluate_log_gradient
        )
        super().__init__(targetDensity, proposalMethod, diagnostics,
                         acceptance=acceptance, rng=rng)

    def _log_mh_ratio(self, transition: TransitionData) -> float:
        """
        Log MH ratio for the Langevin proposal.

        Accounts for the asymmetry of the proposal kernel via the
        quadratic correction term.
        """
        h2 = self._proposalMethod.stepSize ** 2

        x = transition.state.coordinate
        z = transition.proposal.coordinate

        logTarget = float(
            self._tgtDensity.evaluate_log(transition.proposal)
            - self._tgtDensity.evaluate_log(transition.state)
        )

        if logTarget == -np.inf:
            return -np.inf

        # drift at state was pre-computed during generate_proposal
        meanZgivenX = transition.auxiliary['drift']
        meanXgivenZ = self._proposalMethod._drift(transition.proposal)

        diffX = x - meanXgivenZ
        diffZ = z - meanZgivenX

        quadDiff = -0.5 / h2 * (diffX @ diffX - diffZ @ diffZ)

        return logTarget + quadDiff


class MALAFactory(MHFactory):
    """Factory for constructing 'MetropolisAdjustedLangevinAlgorithm' instances."""

    def __init__(self):
        super().__init__()
        self._stepSize = None
        self._acceptance: AcceptanceProbability = None

    @property
    def stepSize(self) -> float:
        return self._stepSize

    @stepSize.setter
    def stepSize(self, h: float):
        if not np.isscalar(h):
            raise TypeError("MALA stepSize must be a scalar.")
        self._stepSize = float(h)

    @property
    def acceptance(self) -> AcceptanceProbability:
        return self._acceptance

    @acceptance.setter
    def acceptance(self, strategy: AcceptanceProbability):
        self._acceptance = strategy

    def _validate(self) -> None:
        super()._validate()

        if self._stepSize is None:
            raise ValueError("MALA stepSize not set.")
        if not np.isfinite(self._stepSize):
            raise ValueError("MALA stepSize must be finite.")
        if self._stepSize <= 0.0:
            raise ValueError(f"Invalid MALA stepSize: {self._stepSize}")

    def _create_sampler(self) -> MetropolisAdjustedLangevinAlgorithm:
        return MetropolisAdjustedLangevinAlgorithm(
            self._target, self._stepSize, self._diagnostics, self._acceptance,
            rng=self.rng
        )
