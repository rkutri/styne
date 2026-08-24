from __future__ import annotations

import numpy as np
from numpy.random import Generator
from styne.parameter.function import Function
from styne.model.representation.expansion import LinearExpansion
from styne.statistics.gaussian import Gaussian
from styne.statistics.interface import CovarianceFunctionInterface
from styne.statistics.measure import ProbabilityMeasure
from styne.utility.grid import Grid

from styne.gp.direct import _DirectGPSpecification
from styne.gp.bspline import _BSplineGPSpecification
from styne.gp.dna import _DNAGPSpecification


class GPSampler(ProbabilityMeasure):

    def __init__(self, expansion: LinearExpansion, measure: Gaussian):

        self._expansion = expansion
        self._measure = measure

    def sample(self, randomState) -> tuple[Function, object]:

        sample, nextState = self._measure.sample(randomState)
        return Function(sample.coordinate, self._expansion), nextState

    def draw(self, rng: Generator) -> Function:
        return self.sample(rng)[0]


class GaussianProcess:
    """
    A Gaussian process defined by a covariance function and linear expansion.

    `parameter` exposes the current field as a `Function` parameter.
    `sampler` returns a `GPSampler` for drawing independent functions.
    """

    def __init__(self, covFcn, specification):
        self._specification = specification
        measureCov, self._expansion = specification.build(covFcn)
        if not isinstance(self._expansion, LinearExpansion):
            raise TypeError("GaussianProcess requires a LinearExpansion")
        self._param = Function(
            np.zeros(self._expansion.dimension), self._expansion
        )

        self._measure = Gaussian(measureCov, self._param)

        self._covFcn = covFcn

    # ---- Explicit builders for supported parametrisations ----

    @classmethod
    def direct(cls, grid: Grid, covFcn: CovarianceFunctionInterface, nugget: float = 0.0) -> GaussianProcess:
        return cls(covFcn, _DirectGPSpecification(grid, nugget))

    @classmethod
    def bspline(
            cls, covFcn: CovarianceFunctionInterface,
            expansion: LinearExpansion) -> GaussianProcess:
        return cls(covFcn, _BSplineGPSpecification(expansion))

    @classmethod
    def dna(cls, covFcn:CovarianceFunctionInterface, q: int | tuple, d: int, alpha: float | tuple = 1.0) -> GaussianProcess:
        return cls(covFcn, _DNAGPSpecification(q, d, alpha))

    @classmethod
    def spde(cls, _) -> GaussianProcess:
        raise NotImplementedError("SPDE parametrisation migration deferred.")

    # ---- Properties and setters ----

    @property
    def spatialDimension(self) -> int:
        return self._specification.spatialDimension

    @property
    def parameterDimension(self) -> int:
        return self._param.dimension

    @property
    def covarianceFunction(self) -> CovarianceFunctionInterface:
        return self._covFcn

    @covarianceFunction.setter
    def covarianceFunction(self, covFcn: CovarianceFunctionInterface) -> None:

        self._covFcn = covFcn
        covariance, expansion = self._specification.build(covFcn)
        self._measure = self._measure.with_covariance(covariance)
        self._replace_expansion(expansion)

    @property
    def parameter(self) -> Function:
        return self._param

    @property
    def expansion(self) -> LinearExpansion:
        return self._expansion

    @property
    def resolution(self) -> tuple:
        if hasattr(self._expansion, 'resolution'):
            return self._expansion.resolution
        raise AttributeError(
            "Current GaussianProcess expansion has no native resolution."
        )

    @property
    def nativeGrid(self):
        if hasattr(self._expansion, 'nativeGrid'):
            return self._expansion.nativeGrid
        if hasattr(self._expansion, 'grid'):
            return self._expansion.grid
        raise AttributeError(
            "Current GaussianProcess expansion has no native grid."
        )

    @property
    def measure(self) -> Gaussian:
        return self._measure

    @property
    def sampler(self) -> GPSampler:
        return GPSampler(self._expansion, self._measure)

    def _replace_expansion(self, expansion: LinearExpansion) -> None:
        coordinate = self._param.coordinate
        meanCoordinate = self._measure.mean.coordinate
        self._expansion = expansion
        self._param = Function(coordinate, expansion)
        self._measure = self._measure.with_mean(Function(meanCoordinate, expansion))

    def bind(self, grid):
        """Return the cached expansion evaluator for ``grid``."""
        return self._expansion.bind(grid)

    def evaluate(self, coefficient, grid):
        """Evaluate explicit GP coordinates on ``grid``."""
        return self.bind(grid).evaluate(coefficient)

    def directional_derivative(self, coefficient, direction, grid):
        """Apply the field derivative to ``direction`` at ``coefficient``."""
        return self.bind(grid).directional_derivative(
            coefficient, direction
        )

    def adjoint_derivative(self, coefficient, cotangent, grid):
        """Pull a field cotangent back to GP coordinate space."""
        return self.bind(grid).adjoint_derivative(
            coefficient, cotangent
        )

    @property
    def hasHyperGradient(self) -> bool:
        return hasattr(self._specification, "evaluate_hyper_gradient")

    @property
    def hasLogLengthMultiplier(self) -> bool:
        return hasattr(
            self._specification, "compute_log_length_multiplier"
        )

    def create_predictor(
            self, coefficient, grid, observationGrid=None):
        return self._specification.create_predictor(
            self, grid, coefficient, observationGrid
        )

    def evaluate_exact_conditional(self, queryGrid, state, sites):
        if not hasattr(
                self._specification, "evaluate_exact_conditional"):
            raise NotImplementedError(
                "Exact conditional evaluation is unavailable for "
                f"{type(self._expansion).__name__}."
            )
        return self._specification.evaluate_exact_conditional(
            self._expansion, queryGrid, state, self._covFcn, sites
        )

    def evaluate_hyper_gradient(self, state, cotangent):
        if not self.hasHyperGradient:
            raise NotImplementedError(
                "Hyperparameter gradients are unavailable for "
                f"{type(self._expansion).__name__}."
            )
        return self._specification.evaluate_hyper_gradient(
            self._expansion, state, cotangent, self._covFcn
        )

    def compute_log_length_multiplier(self, smoothness, lengthScale):
        if not self.hasLogLengthMultiplier:
            raise NotImplementedError(
                "Log-length multipliers are unavailable for "
                f"{type(self._expansion).__name__}."
            )
        return self._specification.compute_log_length_multiplier(
            smoothness, lengthScale
        )
