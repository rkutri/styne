from typing import Optional

from numpy.random import Generator

from styne.backend import infer_backend
from styne.mcmc.method.dart import GaussianDARTProposal
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.diagnostics import ChainDiagnostics
from styne.mcmc.acceptance import AcceptanceProbability
from styne.statistics.gaussian import Gaussian
from styne.statistics.interface import DensityInterface
from styne.statistics.covariance import DenseCovarianceMatrix


class DirectDARTProposal(GaussianDARTProposal):
    """Legacy constructor for the exact Gaussian local-law proposal.

    The supplied covariance must equal (tempering C^-1 + gamma I)^-1.
    New code uses GaussianDARTProposal, which constructs that covariance.
    """

    def __init__(self, tempering, gamma, surrogate, proposalCovariance):
        if not 0 < tempering <= 1 or gamma <= 0:
            raise ValueError("Require tempering in (0, 1] and positive gamma.")
        approximation = Gaussian(surrogate.covariance.with_scaling(
            surrogate.covariance.scaling / tempering
        ), surrogate.mean)
        super().__init__(approximation, gamma)
        supplied = proposalCovariance.scaling * proposalCovariance.to_dense()
        backend = infer_backend(supplied)
        metadata = backend.metadata(supplied)
        expected = self._localised._rootCovariance.apply(backend.eye(
            supplied.shape[-1], dtype=metadata.dtype, device=metadata.device
        ))
        namespace = backend.namespace
        tolerance = 1e-5 if '32' in str(expected.dtype) else 1e-10
        if not bool(namespace.all(namespace.abs(expected - supplied)
                                  <= tolerance * (1 + namespace.abs(expected)))):
            raise ValueError(
                "proposalCovariance is inconsistent with the Gaussian local law."
            )
        # Valid legacy covariance objects retain their factor/noise convention.
        self._localised._rootCovariance = proposalCovariance
        self._localised._reference = self._localised._root_reference()


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
