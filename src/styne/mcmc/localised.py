import copy

from styne.statistics.measure import ProbabilityMeasure
from typing import Optional
import numpy as np

from styne.statistics.interface import DensityInterface
from styne.statistics.gaussian import Gaussian
from styne.statistics.dirac import DiracMeasure
from styne.statistics.covariance import (
    DiagonalCovarianceMatrix, IIDCovarianceMatrix
)
from styne.statistics.radonnikodym import RadonNikodym
from styne.parameter.parameter import Parameter
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.surrogate import SurrogateTransitionMeasure
from styne.utility.densityarithmetic import LogScalingWrapper, ProductWrapper
from styne.parameter.vector import Vector


class LocalisedSurrogateDensity(RadonNikodym):
    """
    Surrogate density localised to a point x in R^d.

    The log-density is

        theta * log pi_k(u) - gamma/2 * ||u - x||^2

    Parameters
    ----------
    regularisation : float
        Regularisation strength gamma (positive).
    tempering : float
        Tempering parameter theta in (0, 1].
    surrogateDensity : DensityInterface
        The surrogate density pi_k.
    temperFullDensity : bool, optional
        If True, temper the full surrogate density (including reference/prior).
        If False (default), temper only the Radon-Nikodym derivative.
    spectralWeights : ndarray, optional
        Per-mode weights for physical-space regularisation. When
        provided, the penalty becomes gamma/2 * ||W(u - x)||^2
        where W = diag(spectralWeights). When None, uses ||u - x||^2.
    """

    def __init__(self,
        regularisation: float,
        tempering: float,
        surrogateDensity: DensityInterface,
        spectralWeights=None,
        temperFullDensity: bool = False
    ) -> None:

        self._surrogateDensity = surrogateDensity

        if regularisation <= 0:
            raise ValueError("Regularisation must be positive.")
        if tempering <= 0. or 1. < tempering:
            raise ValueError("Tempering must be in (0, 1].")

        self._reg = regularisation
        self._tempering = tempering
        self._spectralWeights = spectralWeights
        self._temperFullDensity = temperFullDensity
        self._scaledDerivative = None

        regCov = self._build_reg_covariance(
            surrogateDensity.domainDimension, spectralWeights
        )
        self._regGaussian = Gaussian(regCov)

        if hasattr(surrogateDensity, 'reference'):
            self._regGaussian = self._regGaussian.with_mean(
                surrogateDensity.reference.mean.with_coordinate(
                    np.zeros(surrogateDensity.domainDimension)
                )
            )
        else:
            self._regGaussian = self._regGaussian.with_mean(Vector(
                np.zeros(surrogateDensity.domainDimension)
            ))

        # if the surrogate density is itself a Radon-Nikodym density, there is a
        # choice in which to consider the reference measure. We choose the
        # inherent reference measure of the surrogate density. Typically this will
        # be the prior in a Bayesian model. Important for preconditioned MCMCs.
        if isinstance(surrogateDensity, RadonNikodym):
            scaledDerivative = LogScalingWrapper(surrogateDensity.derivative, tempering)
            self._scaledDerivative = scaledDerivative

            if temperFullDensity:
                scaledCov = surrogateDensity.reference.covariance.with_scaling(
                    surrogateDensity.reference.covariance.scaling / tempering
                )
                reference = Gaussian(scaledCov, surrogateDensity.reference.mean)
                
                self._surrogateComponent = ProductWrapper([
                    LogScalingWrapper(surrogateDensity.reference.density, tempering),
                    scaledDerivative
                ])
            else:
                reference = surrogateDensity.reference
                self._surrogateComponent = ProductWrapper([
                    surrogateDensity.reference.density,
                    scaledDerivative
                ])

            derivative = ProductWrapper([
                self._regGaussian.density,
                scaledDerivative
            ])

        else:
            reference = self._regGaussian
            derivative = LogScalingWrapper(surrogateDensity, tempering)
            self._surrogateComponent = LogScalingWrapper(surrogateDensity, tempering)

        super().__init__(reference, derivative)

    def evaluate_log_surrogate(self, y: Parameter) -> float:
        """Log-density of the tempered surrogate, excluding regularisation.

        In the DART acceptance ratio, the regularisation terms cancel by
        symmetry. Only this component contributes to the outer MH ratio.
        """
        return self._surrogateComponent.evaluate_log(y)

    @property
    def location(self) -> Parameter:
        return self._regGaussian.mean

    @location.setter
    def location(self, location: Parameter):
        self._replace_regularisation(self._regGaussian.with_mean(location))

    def with_location(self, location: Parameter):
        """Return this density localised at ``location``."""
        result = copy.copy(self)
        result.location = location
        return result

    @property
    def regularisation(self) -> float:
        return self._reg

    @property
    def tempering(self) -> float:
        return self._tempering

    @property
    def spectralWeights(self):
        return self._spectralWeights

    def _build_reg_covariance(self, dimension, spectralWeights):
        if spectralWeights is not None:
            return DiagonalCovarianceMatrix(
                1.0 / np.clip(self._reg * spectralWeights**2, 1e-30, None)
            )
        return IIDCovarianceMatrix(dimension, 1.0 / self._reg)

    def sync_weights(self, weights):
        """Rebuild regularisation covariance from updated spectral weights."""
        if weights is None:
            return
        self._spectralWeights = weights
        self._replace_regularisation(
            self._regGaussian.with_covariance(DiagonalCovarianceMatrix(
                1.0 / np.clip(self._reg * weights**2, 1e-30, None)
            ))
        )

    def _replace_regularisation(self, measure):
        self._regGaussian = measure
        if isinstance(self._surrogateDensity, RadonNikodym):
            self._derivative = ProductWrapper([
                measure.density, self._scaledDerivative
            ])
        else:
            self._reference = measure



class LocalisedSurrogateTransitionMeasure(SurrogateTransitionMeasure):
    """
    Surrogate transition measure whose target is a localised density.

    On each draw, the localisation centre and the Dirac initial measure
    are both moved to the current fine-level state before running the
    surrogate chain.

    Parameters
    ----------
    surrogateChain : MetropolisHastings
        Must target a 'LocalisedSurrogateDensity'.
    nChain : int
        Length of each surrogate chain run.
    """

    def __init__(self,
        surrogateChain: MetropolisHastings,
        nChain: int,
        initialMeasure: Optional[ProbabilityMeasure] = None
    ):

        if not isinstance(surrogateChain.target, LocalisedSurrogateDensity):
            raise TypeError("surrogateChain target must be a "
                            "LocalisedSurrogateDensity instance.")

        super().__init__(surrogateChain, nChain, initialMeasure)

    @property
    def location(self) -> Parameter:
        return self._mcmc.target.location

    @location.setter
    def location(self, location: Parameter):
        self._initialMeasure = self._localise_initial_measure(location)
        self._mcmc.target.location = location

    def _localise_initial_measure(self, location: Parameter):
        if hasattr(self._initialMeasure, "with_location"):
            return self._initialMeasure.with_location(location)
        if hasattr(self._initialMeasure, "with_mean"):
            return self._initialMeasure.with_mean(location)

        initialMeasure = copy.copy(self._initialMeasure)
        initialMeasure.location = location
        return initialMeasure

    def transition_trajectory(self, initialState: Parameter, randomState):
        """Run an isolated surrogate trajectory localised at ``initialState``."""
        measure = copy.copy(self)
        measure._mcmc = copy.copy(self._mcmc)
        measure._mcmc._chain = copy.copy(self._mcmc._chain)
        measure._mcmc._diagnostics = copy.deepcopy(self._mcmc.diagnostics)
        measure._mcmc.target = self.density.with_location(initialState)
        measure._initialMeasure = self._localise_initial_measure(initialState)
        trajectoryStart, randomState = measure._initialMeasure.sample(randomState)
        return super(
            LocalisedSurrogateTransitionMeasure, measure
        ).transition_trajectory(trajectoryStart, randomState)

    @property
    def regularisation(self) -> float:
        return self.density.regularisation

    @property
    def tempering(self) -> float:
        return self.density.tempering
