import numpy as np
from numpy import sqrt
from typing import Optional
from numpy.random import Generator

from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.factory import MHFactory
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.transition import TransitionData
from styne.parameter.parameter import Parameter
from styne.statistics.gaussian import Gaussian
from styne.statistics.radonnikodym import RadonNikodym


def _validate_pmala_target(target) -> None:
    if not isinstance(target, RadonNikodym):
        raise TypeError("pMALA target must be a RadonNikodym instance.")
    if not isinstance(target.reference, Gaussian):
        raise NotImplementedError(
            "Currently, only Gaussian reference measures are supported.")
    if not callable(getattr(target.derivative, "evaluate_log_gradient", None)):
        raise ValueError(
            "pMALA requires target.derivative to support evaluate_log_gradient.")


def _validate_beta(beta) -> None:
    if not (0.0 < beta <= 1.0):
        raise ValueError(
            f"Step size must satisfy 0 < beta <= 1. Got {beta}.")


class PMALAProposal(ProposalMethod):
    """
    Preconditioned MALA proposal.

    Extends pCN by adding a Langevin drift (β²/2) C ∇ log Ψ(x), where
    ∇ log Ψ is the gradient of the log Radon-Nikodym derivative. The
    proposal distribution is N(drift(x), β²C).

    The reference covariance is shared with the target (no copy). During
    each draw the scaling is temporarily set to β² so that Gaussian
    sampling produces noise from N(0, β²C), then restored immediately.
    """

    def __init__(self, target: RadonNikodym, beta: float):

        super().__init__()
        _validate_pmala_target(target)
        _validate_beta(beta)

        self._beta = beta
        self._target = target

        self._proposalMeasure = Gaussian(self._target.reference.covariance)

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def referenceMeasure(self) -> Gaussian:
        return self._target.reference

    def _drift(self, state: Parameter) -> np.ndarray:
        x = state.coordinate
        m = self._target.reference.mean.coordinate
        refCov = self._target.reference.covariance
        pcnDrift = m + np.sqrt(1. - self._beta**2) * (x - m)

        try:
            gradLogPsi = (
                self._target.derivative.evaluate_log_gradient(state)
            )
        except (np.linalg.LinAlgError, RuntimeError):
            return pcnDrift

        if not np.all(np.isfinite(gradLogPsi)):
            return pcnDrift

        return pcnDrift + 0.5 * self._beta**2 * refCov.apply(gradLogPsi)


    def generate_proposal(self, rng: Generator) -> TransitionData:
        if self._state is None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        # Drift must be computed before scaling is changed: _drift calls
        # refCov.apply, which uses the current scaling. Setting scaling = β²
        # first would double-count the factor in the drift term.
        driftVector = self._drift(self._state)
        drift = self._state.with_coordinate(
            np.asarray(driftVector, dtype=np.float64)
        )
        refCov = self._target.reference.covariance
        proposalCovariance = refCov.with_scaling(self._beta**2)
        proposalMeasure = self._proposalMeasure.with_mean(drift).with_covariance(
            proposalCovariance
        )
        proposal = proposalMeasure.generate_realisation(rng=rng)

        return TransitionData(
            self._state, proposal, auxiliary={'drift': driftVector}
        )


class PreconditionedMALA(MetropolisHastings):
    r"""
    Preconditioned MALA sampler.

    Extends pCN with a Langevin drift driven by $\nabla \log \Psi$ (the
    derivative's log-gradient). The full target (derivative + reference)
    enters the log acceptance ratio, the reference geometry enters the
    drift and the metric for the quadratic correction term.

    Parameters
    ----------
    target : RadonNikodym
        Target density with Gaussian reference measure.
    beta : float
        Step size in (0, 1].
    diagnostics : ChainDiagnostics
        Tracks transition statistics.
    """
    name = "pMALA"

    def __init__(self, target: RadonNikodym, beta: float, diagnostics,
                 acceptance: AcceptanceProbability = None,
                 rng: Optional[Generator] = None):

        _validate_pmala_target(target)
        _validate_beta(beta)

        proposalMethod = PMALAProposal(target, beta)
        super().__init__(target, proposalMethod, diagnostics,
                         acceptance=acceptance, rng=rng)

    def _log_mh_ratio(self, transition: TransitionData) -> float:

        beta2 = self._proposalMethod.beta**2
        refCov = self._proposalMethod.referenceMeasure.covariance

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

        quadDiff = -0.5 / beta2 * (
            refCov.dual_quadratic_form(diffX)
            - refCov.dual_quadratic_form(diffZ)
        )

        if not np.isfinite(quadDiff):
            return -np.inf

        return logTarget + quadDiff


class PMALAFactory(MHFactory):
    """Factory for constructing PreconditionedMALA instances."""

    def __init__(self):
        super().__init__()
        self._beta: float = None
        self._acceptance: AcceptanceProbability = None

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float):
        self._beta = beta

    @property
    def acceptance(self) -> AcceptanceProbability:
        return self._acceptance

    @acceptance.setter
    def acceptance(self, strategy: AcceptanceProbability):
        self._acceptance = strategy

    def _validate(self) -> None:
        super()._validate()
        _validate_pmala_target(self._target)
        if self._beta is None:
            raise ValueError("Step size parameter (beta) not set for pMALA.")
        _validate_beta(self._beta)

    def _create_sampler(self) -> PreconditionedMALA:
        return PreconditionedMALA(
            self._target, self._beta, self._diagnostics, self._acceptance,
            rng=self.rng
        )
