from typing import Optional

import numpy as np
from numpy.random import Generator

from styne.backend import infer_backend
from styne.parameter.parameter import Parameter
from styne.mcmc.diagnostics import ChainDiagnostics
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.factory import MHFactory
from styne.mcmc.transition import (
    EvaluatedState, RobbinsMonroState, TransitionData,
)
from styne.statistics.interface import DensityInterface
from styne.statistics.covariance import CovarianceMatrix, IIDCovarianceMatrix
from styne.statistics.gaussian import Gaussian
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
        self._proposalMeasure = Gaussian(proposalCov)

    @property
    def covariance(self) -> CovarianceMatrix:
        """Current proposal covariance."""
        return self._proposalMeasure.covariance

    @covariance.setter
    def covariance(self, cov: CovarianceMatrix):
        """Replace the proposal covariance."""
        self._proposalMeasure = self._proposalMeasure.with_covariance(cov)

    def propose(self, state: Parameter, rng):
        return self.propose_with_covariance(state, self.covariance, rng)

    @staticmethod
    def propose_with_covariance(state, covariance, rng):
        backend = infer_backend(state.coordinate)
        metadata = backend.metadata(state.coordinate)
        noise, nextRng = backend.normal(
            rng, state.coordinate.shape,
            dtype=metadata.dtype, device=metadata.device,
        )
        proposal = state.with_coordinate(
            state.coordinate + covariance.apply_chol_factor(noise)
        )
        return TransitionData(state, proposal), nextRng


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
            self, transition: TransitionData):

        return transition.proposed.logDensity - transition.current.logDensity


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
        
        super().__init__(
            target, proposalCov, diagnostics, acceptance=acceptance, rng=rng
        )
        
        if not isinstance(proposalCov, IIDCovarianceMatrix):
            raise TypeError(
                "RobbinsMonroMRW requires an IIDCovarianceMatrix proposal.")
            
        self._targetAcceptance = targetAcceptance
        self._adaptOffset = adaptOffset
        self._adaptDecay = adaptDecay
        
        self._initialVariance = proposalCov.marginalVariance[0]

    def initial_state(self, parameter):
        evaluatedState = super().initial_state(parameter)
        backend = infer_backend(parameter.coordinate)
        metadata = backend.metadata(parameter.coordinate)
        variance = backend.asarray(
            self._initialVariance, dtype=metadata.dtype, device=metadata.device
        )
        return RobbinsMonroState(
            evaluatedState,
            backend.namespace.log(variance),
            backend.asarray(0, dtype=metadata.dtype, device=metadata.device),
        )

    def step(self, currentState, rng):
        backend = infer_backend(currentState.parameter.coordinate)
        variance = backend.namespace.exp(currentState.logVariance)
        covariance = IIDCovarianceMatrix(
            currentState.parameter.dimension, variance
        )
        proposedTransition, proposalRng = self._proposalMethod \
            .propose_with_covariance(
                currentState.parameter, covariance, rng
            )
        proposedState = EvaluatedState(
            proposedTransition.proposed.parameter,
            self._evaluate_log_density(proposedTransition.proposed.parameter),
        )
        transition = TransitionData(
            current=currentState.evaluatedState,
            proposed=proposedState,
            auxiliary=proposedTransition.auxiliary,
        )
        logAcceptanceProbability = self._acceptance.log_probability(
            self._log_mh_ratio(transition)
        )
        metadata = backend.metadata(currentState.parameter.coordinate)
        acceptanceUniform, nextRng = backend.uniform(
            proposalRng, (), dtype=metadata.dtype, device=metadata.device
        )
        outcome = backend.namespace.log(acceptanceUniform) \
            <= logAcceptanceProbability
        transition = TransitionData(
            current=currentState.evaluatedState,
            proposed=proposedState,
            outcome=outcome,
            logAcceptanceProbability=logAcceptanceProbability,
            auxiliary=proposedTransition.auxiliary,
        )
        nextCoordinate = backend.namespace.where(
            outcome,
            proposedState.parameter.coordinate,
            currentState.parameter.coordinate,
        )
        nextDensity = backend.namespace.where(
            outcome,
            proposedState.logDensity,
            currentState.evaluatedState.logDensity,
        )
        nextStepCount = currentState.stepCount + 1
        gamma = 1.0 / (
            nextStepCount + self._adaptOffset
        ) ** self._adaptDecay
        nextLogVariance = currentState.logVariance + gamma * (
            outcome - self._targetAcceptance
        )
        return RobbinsMonroState(
            EvaluatedState(
                currentState.parameter.with_coordinate(nextCoordinate),
                nextDensity,
            ),
            nextLogVariance,
            nextStepCount,
        ), transition, nextRng

    def _record_transition(self, transitionData, nextState):
        super()._record_transition(transitionData, nextState)
        self.proposalCovariance = IIDCovarianceMatrix(
            nextState.parameter.dimension,
            infer_backend(nextState.logVariance).namespace.exp(
                nextState.logVariance
            ),
        )
        
    def clear(self) -> None:
        """
        Reset the chain and revert the proposal covariance to its initial
        variance, discarding all adaptation.

        Returns
        -------
        None
        """
        super().clear()
        dimension = self._proposalMethod.covariance.dimension
        self.proposalCovariance = IIDCovarianceMatrix(
            dimension, self._initialVariance
        )


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
