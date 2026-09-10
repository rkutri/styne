from typing import Optional
from numpy.random import Generator

import numpy as np

from styne.backend import infer_backend
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.factory import MHFactory
from styne.mcmc.transition import TransitionData
from styne.parameter.parameter import Parameter


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

    def __init__(
            self, dim, stepSize, logGradientCallable=None,
            logDensityCallable=None):
        if logGradientCallable is not None \
                and not callable(logGradientCallable):
            raise TypeError("MALA gradient must be callable.")
        if logGradientCallable is None and not callable(logDensityCallable):
            raise TypeError(
                "MALA requires a log density or explicit gradient."
            )
        self._h = float(stepSize)
        self._h2 = self._h * self._h
        self._logGradient = logGradientCallable
        self._logDensity = logDensityCallable

    @property
    def stepSize(self) -> float:
        return self._h

    def _gradient(self, state):
        if self._logGradient is not None:
            return self._logGradient(state)
        backend = infer_backend(state.coordinate)
        if not backend.capabilities.automaticDifferentiation:
            raise ValueError("MALA requires gradient for the NumPy backend.")
        namespace = backend.namespace
        return backend.grad(
            lambda coordinate: namespace.sum(
                self._logDensity(state.with_coordinate(coordinate))
            )
        )(state.coordinate)

    def _drift(self, state: Parameter):
        """Compute the deterministic drift: x + (h^2 / 2) * grad log pi(x)."""
        return state.coordinate + 0.5 * self._h2 * self._gradient(state)

    def propose(self, state: Parameter, rng):
        driftVector = self._drift(state)
        backend = infer_backend(state.coordinate)
        metadata = backend.metadata(state.coordinate)
        noise, nextRng = backend.normal(
            rng, state.coordinate.shape,
            dtype=metadata.dtype, device=metadata.device,
        )
        proposal = state.with_coordinate(
            driftVector + self._h * noise
        )
        return (
            TransitionData(state, proposal, auxiliary={'drift': driftVector}),
            nextRng,
        )


class MetropolisAdjustedLangevinAlgorithm(MetropolisHastings):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA).

    Uses a Langevin-diffusion-inspired proposal with a Metropolis-Hastings
    correction to ensure the correct stationary distribution.

    Parameters
    ----------
    targetDensity : DensityInterface
        Target density. JAX and PyTorch differentiate 'evaluate_log'.
        NumPy execution requires an explicit 'gradient' callable.
    stepSize : float
        Step size h (standard deviation of the isotropic noise).
    diagnostics : ChainDiagnostics
        Tracks transition statistics.
    """
    name = "MALA"

    def __init__(self, targetDensity, stepSize, diagnostics,
                 acceptance: AcceptanceProbability = None,
                 rng: Optional[Generator] = None, gradient=None):

        proposalMethod = MALAProposal(
            targetDensity.domainDimension,
            stepSize,
            gradient,
            targetDensity.evaluate_log,
        )
        super().__init__(targetDensity, proposalMethod, diagnostics,
                         acceptance=acceptance, rng=rng)

    def _proposal_for_target(self, targetDensity):
        gradient = self._proposalMethod._logGradient
        owner = getattr(gradient, "__self__", None)
        if owner is self.target:
            gradient = targetDensity.evaluate_log_gradient
        return MALAProposal(
            targetDensity.domainDimension,
            self._proposalMethod.stepSize,
            gradient,
            targetDensity.evaluate_log,
        )

    def _log_mh_ratio(self, transition: TransitionData):
        """
        Log MH ratio for the Langevin proposal.

        Accounts for the asymmetry of the proposal kernel via the
        quadratic correction term.
        """
        h2 = self._proposalMethod.stepSize ** 2

        x = transition.state.coordinate
        z = transition.proposal.coordinate

        logTarget = (
            transition.proposed.logDensity - transition.current.logDensity
        )

        meanZgivenX = transition.auxiliary['drift']
        meanXgivenZ = self._proposalMethod._drift(transition.proposal)

        diffX = x - meanXgivenZ
        diffZ = z - meanZgivenX

        backend = infer_backend(x)
        quadDiff = -0.5 / h2 * (
            backend.namespace.sum(diffX * diffX, axis=-1)
            - backend.namespace.sum(diffZ * diffZ, axis=-1)
        )

        return logTarget + quadDiff


class MALAFactory(MHFactory):
    """Factory for constructing 'MetropolisAdjustedLangevinAlgorithm' instances."""

    def __init__(self):
        super().__init__()
        self._stepSize = None
        self._acceptance: AcceptanceProbability = None
        self._gradient = None

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self, value):
        if value is not None and not callable(value):
            raise TypeError("MALA gradient must be callable.")
        self._gradient = value

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
            rng=self.rng, gradient=self._gradient,
        )
