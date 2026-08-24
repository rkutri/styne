from numpy import sqrt
from typing import Optional
from numpy.random import Generator

from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.transition import TransitionData
from styne.statistics.radonnikodym import RadonNikodym
from styne.statistics.measure import ProbabilityMeasure
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.factory import MHFactory
from styne.parameter.parameter import Parameter
from styne.statistics.gaussian import Gaussian


def validate_beta(beta) -> None:
    if not (0.0 < beta <= 1.0):
        raise ValueError(
            f"pCN step size must satisfy 0 < beta <= 1. Got {beta}.")


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
    target : RadonNikodym
        Target density expressed as a Radon-Nikodym derivative with
        respect to a Gaussian reference measure.
    beta : float
        Step size in (0, 1]. Controls the balance between persistence
        (small beta) and exploration (large beta).
    """

    def __init__(self, referenceMeasure: ProbabilityMeasure, beta: float):

        super().__init__()

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

    def generate_proposal(self, rng: Generator) -> Parameter:
        # Guard against use outside the MH loop, where state may not be set.
        if self._state is None:
            raise ValueError(
                "Trying to generate proposal with undefined state"
            )

        x = self._state.coordinate
        xi = self._refMeasure.generate_realisation(rng=rng).coordinate
        m = self._refMeasure.mean.coordinate

        xCentred = x - m
        xiCentred = xi - m

        zCentred = sqrt(1. - self._beta**2) * xCentred \
            + self._beta * xiCentred

        proposal = self._state.with_coordinate(m + zCentred)

        return TransitionData(self._state, proposal)


class PreconditionedCrankNicolson(MetropolisHastings):
    """
    Preconditioned Crank-Nicolson sampler.

    Since the pCN proposal preserves the reference measure, the log MH
    ratio reduces to the difference in the log Radon-Nikodym derivative
    (the log-likelihood ratio).

    Parameters
    ----------
    target : RadonNikodym
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

        if not isinstance(target, RadonNikodym):
            raise TypeError(
                "pCN target must be a RadonNikodym instance (with a Gaussian "
                "reference measure)."
            )

        proposalMethod = PCNProposal(target.reference, beta)
        super().__init__(target, proposalMethod, diagnostics,
                         acceptance=acceptance, rng=rng)

    def _log_mh_ratio(self, transition: TransitionData) -> float:

        return self._tgtDensity.derivative.evaluate_log(transition.proposal) \
            - self._tgtDensity.derivative.evaluate_log(transition.state)


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

        if not isinstance(self._target, RadonNikodym):
            raise TypeError(
                "pCN target must provide a Gaussian reference "
                "and RN-derivative implementation."
            )
        if self._beta is None:
            raise ValueError("Step size parameter (beta) not set for pCN.")
        validate_beta(self._beta)

    def _create_sampler(self) -> PreconditionedCrankNicolson:
        return PreconditionedCrankNicolson(
            self._target, self._beta, self._diagnostics, self._acceptance,
            rng=self.rng
        )
