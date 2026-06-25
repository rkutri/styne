from numpy import ndarray, asarray
from numpy.random import Generator

from styne.statistics.interface import CovarianceFunctionInterface
from styne.statistics.covariance import DenseCovarianceMatrix
from styne.statistics.measure import ProbabilityMeasure


class DenseGPEngine(ProbabilityMeasure):

    def __init__(self, grid, covarianceFunction: CovarianceFunctionInterface):

        if not isinstance(covarianceFunction, CovarianceFunctionInterface):
            raise TypeError("Invalid covariance function type.")

        grid = asarray(grid, dtype=float)

        if grid.ndim == 1:
            grid = grid[:, None]

        if grid.ndim != 2:
            raise ValueError("grid must be array of shape (nPoints, dim).")

        self._grid = grid
        self._covFcn = covarianceFunction

        self._covMat = self._covFcn.evaluate_covariance(self._grid, self._grid)

    def draw(self, rng: Generator) -> ndarray:
        return self._covMat.apply_chol_factor(
            rng.standard_normal(self._grid.shape[0])
        )
