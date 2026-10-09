from typing import Optional
from numpy.random import Generator

from styne.backend import infer_backend
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.factory import MHFactory
from styne.mcmc.proposal import ProposalMethod
from styne.parameter.parameter import Parameter
from styne.statistics.gaussian import Gaussian
from styne.statistics.interface import RadonNikodymInterface


def validate_pmala_target(target) -> None:
    if not isinstance(target, RadonNikodymInterface):
        raise TypeError(
            "pMALA target must implement RadonNikodymInterface."
        )
    if not isinstance(target.reference, Gaussian):
        raise NotImplementedError(
            "Currently, only Gaussian reference measures are supported.")


def validate_beta(beta) -> None:
    if not (0.0 < beta <= 1.0):
        raise ValueError(
            f"Step size must satisfy 0 < beta <= 1. Got {beta}.")


class PMALAProposal(ProposalMethod):
    """
    Preconditioned MALA proposal.

    Extends pCN by adding a Langevin drift (β²/2) C ∇ log Ψ(x), where
    ∇ log Ψ is the gradient of the log Radon-Nikodym derivative. The
    proposal distribution is N(drift(x), β²C).

    The reference covariance colors backend-native Gaussian noise directly.
    JAX and PyTorch differentiate the derivative density automatically;
    NumPy execution requires an explicit gradient callable. The correction
    compares the proposal densities in both directions relative to the
    Lebesgue measure.
    """

    def __init__(
            self, target: RadonNikodymInterface, beta: float, gradient=None):

        validate_pmala_target(target)
        validate_beta(beta)
        if gradient is not None and not callable(gradient):
            raise TypeError("pMALA gradient must be callable.")
        referenceBackend = infer_backend(target.reference.mean.coordinate)
        if referenceBackend.name == "numpy" and gradient is None:
            raise ValueError("pMALA requires gradient for the NumPy backend.")

        self._beta = beta
        self._target = target
        self._gradient = gradient

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def referenceMeasure(self) -> Gaussian:
        return self._target.reference

    def for_target(self, target, previous):
        if self._target is not previous:
            return self
        validate_pmala_target(target)
        gradient = self._gradient
        owner = getattr(gradient, "__self__", None)
        if owner is previous:
            gradient = target.evaluate_log_gradient
        elif owner is previous.derivative:
            gradient = target.derivative.evaluate_log_gradient
        return type(self)(target, self._beta, gradient)

    def _gradient_at(self, state):
        if self._gradient is not None:
            return self._gradient(state)
        backend = infer_backend(state.coordinate)
        namespace = backend.namespace
        return backend.grad(
            lambda coordinate: namespace.sum(
                self._target.derivative.evaluate_log(
                    state.with_coordinate(coordinate)
                )
            )
        )(state.coordinate)

    def _drift(self, state: Parameter):
        x = state.coordinate
        m = self._target.reference.mean.coordinate
        refCov = self._target.reference.covariance
        backend = infer_backend(x)
        metadata = backend.metadata(x)
        persistence = backend.namespace.sqrt(backend.asarray(
            1. - self._beta ** 2,
            dtype=metadata.dtype, device=metadata.device,
        ))
        pcnDrift = m + persistence * (x - m)
        return pcnDrift + 0.5 * self._beta ** 2 * refCov.apply(
            self._gradient_at(state)
        )

    def propose(self, state: Parameter, rng):
        driftVector = self._drift(state)
        backend = infer_backend(state.coordinate)
        metadata = backend.metadata(state.coordinate)
        noise, nextRng = backend.normal(
            rng, state.coordinate.shape,
            dtype=metadata.dtype, device=metadata.device,
        )
        refCov = self._target.reference.covariance
        proposal = state.with_coordinate(
            driftVector + self._beta * refCov.apply_chol_factor(noise)
        )
        # The forward Gaussian density only needs the drawn noise.
        reverse = state.coordinate - self._drift(proposal)
        logCorrection = (
            0.5 * backend.namespace.sum(noise * noise, axis=-1)
            - 0.5 / self._beta ** 2 * refCov.dual_quadratic_form(reverse)
        )
        return self.record(state, proposal, logCorrection), nextRng


class PreconditionedMALA(MetropolisHastings):
    r"""
    Preconditioned MALA sampler.

    Extends pCN with a Langevin drift driven by $\nabla \log \Psi$ (the
    derivative's log-gradient). The reference geometry enters the drift and
    the metric of the proposal's correction.

    Parameters
    ----------
    target : RadonNikodymInterface
        Target density with Gaussian reference measure.
    beta : float
        Step size in (0, 1].
    diagnostics : ChainDiagnostics
        Tracks transition statistics.
    """
    name = "pMALA"

    def __init__(self, target: RadonNikodymInterface, beta: float, diagnostics,
                 acceptance: AcceptanceProbability = None,
                 rng: Optional[Generator] = None, gradient=None):

        validate_pmala_target(target)
        validate_beta(beta)

        proposalMethod = PMALAProposal(target, beta, gradient)
        super().__init__(target, proposalMethod, diagnostics,
                         acceptance=acceptance, rng=rng)


class PMALAFactory(MHFactory):
    """Factory for constructing PreconditionedMALA instances."""

    def __init__(self):
        super().__init__()
        self._beta: float = None
        self._acceptance: AcceptanceProbability = None
        self._gradient = None

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self, value):
        if value is not None and not callable(value):
            raise TypeError("pMALA gradient must be callable.")
        self._gradient = value

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
        validate_pmala_target(self._target)
        if self._beta is None:
            raise ValueError("Step size parameter (beta) not set for pMALA.")
        validate_beta(self._beta)

    def _create_sampler(self) -> PreconditionedMALA:
        return PreconditionedMALA(
            self._target, self._beta, self._diagnostics, self._acceptance,
            rng=self.rng, gradient=self._gradient,
        )
