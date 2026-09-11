from typing import Optional
from numpy.random import Generator

from styne.backend import infer_backend
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.transition import TransitionData
from styne.statistics.interface import RadonNikodymInterface
from styne.statistics.measure import ProbabilityMeasure
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.factory import MHFactory
from styne.parameter.parameter import Parameter
from styne.statistics.gaussian import Gaussian


def validate_beta(beta) -> None:
    if not (0.0 < beta <= 1.0):
        raise ValueError(
            f"pCN step size must satisfy 0 < beta <= 1. Got {beta}.")


def validate_pcn_target(target) -> None:
    """Validate the RN and Gaussian-reference contract required by pCN."""
    if not isinstance(target, RadonNikodymInterface):
        raise TypeError(
            "pCN target must implement RadonNikodymInterface."
        )
    if not isinstance(target.reference, Gaussian):
        raise NotImplementedError(
            "pCN requires a Gaussian reference measure."
        )


class PCNProposal(ProposalMethod):
    """
    Preconditioned Crank-Nicolson proposal.

    Given step size beta and reference measure N(m, C), the proposal
    from state x is

        z = m + sqrt(1 - beta^2) * (x - m) + beta * (xi - m)

    where xi ~ N(m, C). This preserves the reference measure by
    construction.

    Parameters
    ----------
    referenceMeasure : Gaussian
        Gaussian reference measure preserved by the proposal.
    beta : float
        Step size in (0, 1]. Controls the balance between persistence
        (small beta) and exploration (large beta).
    """

    def __init__(self, referenceMeasure: ProbabilityMeasure, beta: float):

        if not isinstance(referenceMeasure, Gaussian):
            raise NotImplementedError(
                "Currently, only Gaussian reference measures are supported"
            )

        validate_beta(beta)

        self._beta = beta
        self._refMeasure = referenceMeasure

    @property
    def referenceMeasure(self) -> Gaussian:
        return self._refMeasure

    @property
    def beta(self) -> float:
        return self._beta

    def propose(self, state: Parameter, rng):
        coordinate = state.coordinate
        backend = infer_backend(coordinate)
        metadata = backend.metadata(coordinate)
        noise, nextRng = backend.normal(
            rng, coordinate.shape,
            dtype=metadata.dtype, device=metadata.device,
        )
        mean = self._refMeasure.mean.coordinate
        priorNoise = self._refMeasure.covariance.apply_chol_factor(noise)
        persistence = backend.namespace.sqrt(backend.asarray(
            1. - self._beta ** 2,
            dtype=metadata.dtype, device=metadata.device,
        ))
        proposal = state.with_coordinate(
            mean + persistence * (coordinate - mean) + self._beta * priorNoise
        )

        return TransitionData(state, proposal), nextRng


class PreconditionedCrankNicolson(MetropolisHastings):
    """
    Preconditioned Crank-Nicolson sampler.

    Since the pCN proposal preserves the reference measure, the log MH
    ratio reduces to the difference in the log Radon-Nikodym derivative
    (the log-likelihood ratio).

    Parameters
    ----------
    target : RadonNikodymInterface
        Target density with Gaussian reference measure.
    beta : float
        Step size in (0, 1].
    diagnostics : ChainDiagnostics
        Tracks transition statistics.
    """
    name = "pCN"

    def __init__(self, target, beta, diagnostics,
                 acceptance: AcceptanceProbability = None,
                 rng: Optional[Generator] = None):

        validate_pcn_target(target)

        proposalMethod = PCNProposal(target.reference, beta)
        super().__init__(target, proposalMethod, diagnostics,
                         acceptance=acceptance, rng=rng)

    def _evaluate_log_density(self, parameter: Parameter):
        return self._tgtDensity.derivative.evaluate_log(parameter)

    def _proposal_for_target(self, targetDensity):
        validate_pcn_target(targetDensity)
        return PCNProposal(targetDensity.reference, self.proposal.beta)

    def _log_mh_ratio(self, transition: TransitionData):
        return transition.proposed.logDensity - transition.current.logDensity


class PCNFactory(MHFactory):
    """Factory for constructing 'PreconditionedCrankNicolson' instances."""

    def __init__(self):
        super().__init__()
        self._beta = None
        self._acceptance: AcceptanceProbability = None

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def acceptance(self) -> AcceptanceProbability:
        return self._acceptance

    @acceptance.setter
    def acceptance(self, strategy: AcceptanceProbability):
        self._acceptance = strategy

    def _validate(self) -> None:
        super()._validate()

        validate_pcn_target(self._target)
        if self._beta is None:
            raise ValueError("Step size parameter (beta) not set for pCN.")
        validate_beta(self._beta)

    def _create_sampler(self) -> PreconditionedCrankNicolson:
        return PreconditionedCrankNicolson(
            self._target, self._beta, self._diagnostics, self._acceptance,
            rng=self.rng
        )
