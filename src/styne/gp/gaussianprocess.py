from __future__ import annotations

import numpy as np
from numpy.random import Generator
from styne.gp.engine import GPEngine
from styne.parameter.function import Function
from styne.model.representation.expansion import Expansion
from styne.statistics.gaussian import Gaussian
from styne.statistics.interface import CovarianceFunctionInterface
from styne.statistics.measure import ProbabilityMeasure
from styne.utility.grid import Grid

from styne.gp.direct import DirectGPEngine
from styne.gp.bspline import BSplineGPEngine
from styne.gp.dna import DNAFourierEngine


class GPSampler(ProbabilityMeasure):

    def __init__(self, expansion: Expansion, measure: Gaussian):

        self._expansion = expansion
        self._measure = measure

    def draw(self, rng: Generator) -> Function:

        sample = self._measure.draw(rng)
        return Function(sample.coordinate, self._expansion)


class GaussianProcess:
    """
    A Gaussian process defined by a covariance function and a GPEngine.

    The engine handles the parametrisation. `parameter` exposes the
    current field as a `Function` parameter for use in forward models.
    `sampler` returns a `GPSampler` for drawing independent functions.
    """

    def __init__(self, covFcn: CovarianceFunctionInterface, engine: GPEngine):

        measureCov = engine.build_covariance(covFcn)
        self._expansion = engine.build_expansion()
        self._param = Function(
            np.zeros(self._expansion.dimension), self._expansion
        )

        self._measure = Gaussian(measureCov)
        self._measure.mean = self._param.clone()

        self._covFcn = covFcn
        self._engine = engine

        self._sites = None

    # ---- Explicit builders for currently supported engines ----

    @classmethod
    def direct(cls, grid: Grid, covFcn: CovarianceFunctionInterface, nugget: float = 0.0) -> GaussianProcess:
        return cls(covFcn, DirectGPEngine(grid, nugget))

    @classmethod
    def bspline(cls, covFcn: CovarianceFunctionInterface, expansion: Expansion) -> GaussianProcess:
        return cls(covFcn, BSplineGPEngine(expansion))

    @classmethod
    def dna(cls, covFcn:CovarianceFunctionInterface, q: int | tuple, d: int, alpha: float | tuple = 1.0) -> GaussianProcess:
        return cls(covFcn, DNAFourierEngine(q, d, alpha))

    @classmethod
    def spde(cls, _) -> GaussianProcess:
        raise NotImplementedError("SPDE engine migration deferred.")

    # ---- Properties and setters ----

    @property
    def spatialDimension(self) -> int:
        return self._engine.spatialDimension

    @property
    def parameterDimension(self) -> int:
        return self._param.dimension

    @property
    def covarianceFunction(self) -> CovarianceFunctionInterface:
        return self._covFcn

    @covarianceFunction.setter
    def covarianceFunction(self, covFcn: CovarianceFunctionInterface) -> None:

        self._covFcn = covFcn
        self._measure.covariance = self._engine.build_covariance(covFcn)
        self._replace_expansion(self._engine.build_expansion())

    @property
    def parameter(self) -> Function:
        return self._param

    @property
    def expansion(self) -> Expansion:
        return self._expansion

    @property
    def resolution(self) -> tuple:
        if hasattr(self._engine, 'resolution'):
            return self._engine.resolution
        raise AttributeError(
            "Current GaussianProcess engine does not expose a native resolution."
        )

    @property
    def measure(self) -> Gaussian:
        return self._measure

    @property
    def sites(self):
        """
        The spatial locations where the GP is currently evaluated.

        Note
        ----
        Mutating this property frequently (e.g. inside a tight MCMC loop)
        carries a significant performance penalty. For 'DirectGPEngine',
        it triggers a full $O(N^3)$ Cholesky factorisation. For engines with
        internal interpolation mapping like 'DNAFourierEngine', it triggers
        a full rebuild of the interpolation matrices.
        
        To evaluate the GP at a sequence of different grids without mutating
        its anchor, use the engine's native evaluations combined with
        static interpolation matrices built outside the loop.
        """
        return self._sites

    @sites.setter
    def sites(self, sites: Grid) -> None:

        if not isinstance(sites, Grid):
            raise TypeError("sites must be a Grid")

        self._sites = sites
        self._engine.set_sites(self._sites)

        if self._engine.requires_covariance_rebuild():

            self._measure.covariance = self._engine.build_covariance(self._covFcn)
            self._expansion = self._engine.build_expansion()
            self._param = Function(
                np.zeros(self._expansion.dimension), self._expansion
            )
            self._measure.mean = self._param.clone()

    @property
    def engine(self):
        return self._engine

    @property
    def sampler(self) -> GPSampler:
        return GPSampler(self._expansion, self._measure)

    def _replace_expansion(self, expansion: Expansion) -> None:
        coordinate = self._param.coordinate
        meanCoordinate = self._measure.mean.coordinate
        self._expansion = expansion
        self._param = Function(coordinate, expansion)
        self._measure.mean = Function(meanCoordinate, expansion)

    def at_sites(self, coefficient: np.ndarray) -> np.ndarray:
        """Evaluate an explicit latent coefficient at the configured sites.

        Evaluation is stateless: the coefficient is passed directly through
        the engine's linear synthesis operator and the stored GP parameter is
        not read or mutated.
        """
        if self._sites is None:
            raise ValueError("sites not set on GaussianProcess")

        return self._engine.evaluate(
            coefficient, self._measure.covariance)

    def directional_derivative(self, vector: np.ndarray) -> np.ndarray:
        return self._engine.apply_jacobian(
            vector, self._measure.covariance)

    def adjoint_directional_derivative(
            self, cotangent: np.ndarray) -> np.ndarray:
        return self._engine.apply_adjoint_jacobian(
            cotangent, self._measure.covariance)
