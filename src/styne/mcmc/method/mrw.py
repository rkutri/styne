from styne.parameter.parameter import Parameter
from styne.mcmc.diagnostics import ChainDiagnostics
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.factory import MHFactory
from styne.mcmc.transition import TransitionData
from styne.statistics.interface import DensityInterface
from styne.statistics.covariance import CovarianceMatrix, IIDCovarianceMatrix
from styne.statistics.gaussian import Gaussian
import numpy as np
from typing import Optional
from numpy.random import Generator


class MRWProposal(ProposalMethod):
    """
    Symmetric Gaussian random walk proposal.

    Draws proposals from N(state, proposalCov).

    Parameters
    ----------
    proposalCov : CovarianceMatrix
        Covariance of the proposal distribution.
    """

    def __init__(self, proposalCov: CovarianceMatrix):
        super().__init__()
        self._proposalMeasure = Gaussian(proposalCov)

    @property
    def covariance(self) -> CovarianceMatrix:
        """Current proposal covariance."""
        return self._proposalMeasure.covariance

    @covariance.setter
    def covariance(self, cov: CovarianceMatrix):
        """Replace the proposal covariance."""
        self._proposalMeasure = self._proposalMeasure.with_covariance(cov)

    @ProposalMethod.state.setter
    def state(self, state: Parameter):
        ProposalMethod.state.fset(self, state)
        self._proposalMeasure = self._proposalMeasure.with_mean(state)

    def generate_proposal(self, rng: Generator):
        # Guard against use outside the MH loop, where state may not be set.
        if self._state is None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        return TransitionData(
            self._state, self._proposalMeasure.generate_realisation(rng=rng)
        )


class MetropolisedRandomWalk(MetropolisHastings):
    """
    Metropolis-Hastings with a symmetric Gaussian random walk proposal.

    Since the proposal kernel is symmetric, the log MH ratio reduces to
    the log-density difference between proposal and current state.

    Parameters
    ----------
    target : DensityInterface
        Target density to sample from.
    proposalCov : CovarianceMatrix
        Covariance of the Gaussian proposal kernel.
    diagnostics : ChainDiagnostics
        Tracks transition statistics.
    """
    name = "MRW"

    def __init__(
            self,
            target: DensityInterface,
            proposalCov: CovarianceMatrix,
            diagnostics: ChainDiagnostics,
            acceptance: AcceptanceProbability = None,
            rng: Optional[Generator] = None):

        proposalMethod = MRWProposal(proposalCov)
        super().__init__(target, proposalMethod, diagnostics,
                         acceptance=acceptance, rng=rng)

    @property
    def proposalCovariance(self) -> CovarianceMatrix:
        """Current proposal covariance."""
        return self._proposalMethod.covariance

    @proposalCovariance.setter
    def proposalCovariance(self, cov: CovarianceMatrix):
        """Replace the proposal covariance."""
        self._proposalMethod.covariance = cov

    def _log_mh_ratio(
            self, transition: TransitionData) -> float:

        return self._tgtDensity.evaluate_log(transition.proposal) \
            - self._tgtDensity.evaluate_log(transition.state)


class MRWFactory(MHFactory):
    """Factory for constructing 'MetropolisedRandomWalk' instances."""

    def __init__(self):
        super().__init__()
        self._proposalCov: CovarianceMatrix = None
        self._acceptance: AcceptanceProbability = None

    @property
    def proposalCovariance(self) -> CovarianceMatrix:
        return self._proposalCov

    @proposalCovariance.setter
    def proposalCovariance(self, cov: CovarianceMatrix):
        self._proposalCov = cov

    @property
    def acceptance(self) -> AcceptanceProbability:
        return self._acceptance

    @acceptance.setter
    def acceptance(self, strategy: AcceptanceProbability):
        self._acceptance = strategy

    def _validate(self) -> None:
        super()._validate()
        if self._proposalCov is None:
            raise ValueError("Proposal covariance not set for MRWFactory.")

    def _create_sampler(self) -> MetropolisHastings:
        return MetropolisedRandomWalk(
            self._target, self._proposalCov, self._diagnostics,
            self._acceptance, rng=self.rng)


class RobbinsMonroMRW(MetropolisedRandomWalk):
    """
    Adaptive Metropolis-Hastings with a symmetric Gaussian random walk proposal.

    Uses Robbins-Monro adaptation (Andrieu & Thoms 2008) to tune the proposal
    variance towards a target acceptance rate. The adaptation vanishes over time
    to preserve ergodicity.

    Parameters
    ----------
    target : DensityInterface
        Target density to sample from.
    proposalCov : CovarianceMatrix
        Initial covariance of the Gaussian proposal kernel.
    diagnostics : ChainDiagnostics
        Tracks transition statistics.
    targetAcceptance : float, default 0.3
        Acceptance rate the adaptation targets.
    adaptOffset : int, default 100
        Offset in the vanishing adaptation step-size schedule.
    adaptDecay : float, default 0.6
        Decay exponent in the vanishing adaptation step-size schedule.
    """
    def __init__(
            self,
            target: DensityInterface,
            proposalCov: CovarianceMatrix,
            diagnostics: ChainDiagnostics,
            acceptance: AcceptanceProbability = None,
            targetAcceptance: float = 0.3,
            adaptOffset: int = 100,
            adaptDecay: float = 0.6,
            rng: Optional[Generator] = None):
        
        super().__init__(target, proposalCov, diagnostics, acceptance=acceptance, rng=rng)
        
        if not isinstance(proposalCov, IIDCovarianceMatrix):
            raise TypeError(
                "RobbinsMonroMRW requires an IIDCovarianceMatrix proposal.")
            
        self._targetAcceptance = targetAcceptance
        self._adaptOffset = adaptOffset
        self._adaptDecay = adaptDecay
        
        self._stepCount = 0
        self._logVariance = np.log(proposalCov.marginalVariance[0])
        self._initialLogVariance = self._logVariance

    def _process_transition(self, transitionData):
        nextState = super()._process_transition(transitionData)
        
        alpha = 1.0 if transitionData.outcome == TransitionData.ACCEPTED else 0.0
        
        self._stepCount += 1
        gamma = 1.0 / (self._stepCount + self._adaptOffset) ** self._adaptDecay
        
        self._logVariance += gamma * (alpha - self._targetAcceptance)
        
        dimension = self._proposalMethod.covariance.dimension
        self.proposalCovariance = IIDCovarianceMatrix(
            dimension, np.exp(self._logVariance))
        
        return nextState

    def clear(self) -> None:
        """
        Reset the chain and revert the proposal covariance to its initial
        variance, discarding all adaptation.

        Returns
        -------
        None
        """
        super().clear()
        self._stepCount = 0
        self._logVariance = self._initialLogVariance
        dimension = self._proposalMethod.covariance.dimension
        self.proposalCovariance = IIDCovarianceMatrix(
            dimension, np.exp(self._logVariance))


class RobbinsMonroMRWFactory(MHFactory):
    """Factory for constructing 'RobbinsMonroMRW' instances."""

    def __init__(self):
        super().__init__()
        self._proposalCov: CovarianceMatrix = None
        self._acceptance: AcceptanceProbability = None
        self.targetAcceptance: float = 0.3
        self.adaptOffset: int = 100
        self.adaptDecay: float = 0.6

    @property
    def proposalCovariance(self) -> CovarianceMatrix:
        return self._proposalCov

    @proposalCovariance.setter
    def proposalCovariance(self, cov: CovarianceMatrix):
        self._proposalCov = cov

    @property
    def acceptance(self) -> AcceptanceProbability:
        return self._acceptance

    @acceptance.setter
    def acceptance(self, strategy: AcceptanceProbability):
        self._acceptance = strategy

    def _validate(self) -> None:
        super()._validate()
        if self._proposalCov is None:
            raise ValueError("Proposal covariance not set for RobbinsMonroMRWFactory.")

    def _create_sampler(self) -> MetropolisHastings:
        return RobbinsMonroMRW(
            self._target, self._proposalCov, self._diagnostics,
            self._acceptance, self.targetAcceptance,
            self.adaptOffset, self.adaptDecay, rng=self.rng)
