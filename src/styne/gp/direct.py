import numpy as np
from styne.backend import infer_backend
from styne.model.representation.expansion import (
    BoundLinearExpansion,
    LinearExpansion,
    backend_constant,
)
from styne.statistics.interface import CovarianceFunctionInterface, Predictor
from styne.statistics.covariance import CovarianceMatrix, DenseCovarianceMatrix, IIDCovarianceMatrix
from styne.utility.grid import Grid
from styne.utility.interpolation import linear_interpolation_matrix


class _DirectEvaluation(BoundLinearExpansion):

    def __init__(self, interpolation, shapeFactor=None):
        self._interpolation = np.asarray(interpolation)
        self._shapeFactor = shapeFactor

    @property
    def dimension(self):
        return self._interpolation.shape[1]

    def evaluate(self, coefficient):
        if coefficient.ndim < 1 or coefficient.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected coefficient shape (..., {self.dimension}); "
                f"got {coefficient.shape}."
            )
        native = coefficient
        if self._shapeFactor is not None:
            shapeFactor = backend_constant(
                self._shapeFactor, coefficient
            )
            native = native @ shapeFactor.T
        interpolation = backend_constant(
            self._interpolation, coefficient
        )
        return native @ interpolation.T

    def _adjoint_derivative(self, coefficient, cotangent):
        interpolation = backend_constant(self._interpolation, cotangent)
        nativeCotangent = cotangent @ interpolation
        if self._shapeFactor is None:
            return nativeCotangent
        shapeFactor = backend_constant(self._shapeFactor, cotangent)
        return nativeCotangent @ shapeFactor


class DirectExpansion(LinearExpansion):
    """
    A Gaussian-process parametrisation on a fixed representation grid.

    When ``shapeCovariance`` is present, coordinates are whitened and its
    Cholesky factor maps them to field values on the representation grid.

    Parameters
    ----------
    grid : Grid
        The grid on which the direct coefficients live.
    dimension : int
        Number of coefficients (grid points).
    """

    def __init__(
            self, grid: Grid, dimension: int,
            shapeCovariance: CovarianceMatrix = None):

        self._dim = dimension
        self._grid = grid
        self._shapeCov = shapeCovariance

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def grid(self):
        return self._grid

    @property
    def shapeCovariance(self):
        return self._shapeCov

    def _bind(self, grid: Grid) -> BoundLinearExpansion:
        anchor = self._grid.to_array()
        query = grid.to_array()
        if self._grid.dimension == 1:
            interpolation = linear_interpolation_matrix(
                query.ravel(), anchor.ravel()
            ).toarray()
        elif np.array_equal(query, anchor):
            interpolation = np.eye(self._dim)
        else:
            raise NotImplementedError(
                "DirectExpansion supports off-grid evaluation only in 1D."
            )

        shapeFactor = None if self._shapeCov is None else \
            self._shapeCov.to_cholesky()
        return _DirectEvaluation(interpolation, shapeFactor)


class _DirectGPSpecification:
    """Construction and prediction rules for a direct GP parametrisation."""

    def __init__(self, grid: Grid, nugget: float = 0.0):
        self._grid = grid
        self._nugget = nugget

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def spatialDimension(self) -> int:
        return self._grid.dimension

    def build(self, covarianceFunction: CovarianceFunctionInterface):
        covarianceMatrix = covarianceFunction.evaluate_covariance(
            self._grid, self._grid
        )

        if not isinstance(covarianceMatrix, CovarianceMatrix):
            backend = infer_backend(covarianceMatrix)
            metadata = backend.metadata(covarianceMatrix)
            covarianceMatrix = 0.5 * (covarianceMatrix + covarianceMatrix.T)
            if self._nugget > 0.:
                covarianceMatrix = covarianceMatrix + self._nugget * backend.eye(
                    len(self._grid), dtype=metadata.dtype, device=metadata.device
                )

        shapeCovariance = covarianceMatrix if \
            isinstance(covarianceMatrix, CovarianceMatrix) \
            else DenseCovarianceMatrix(covarianceMatrix)
        expansion = DirectExpansion(
            self._grid, len(self._grid), shapeCovariance
        )
        return IIDCovarianceMatrix(len(self._grid), 1.0), expansion

    def evaluate_exact_conditional(
            self, expansion, queryGrid, state, covFcn, sites):
        z = state.coordinate if hasattr(state, 'coordinate') else state
        backend = infer_backend(z)
        metadata = backend.metadata(z)
        observed = expansion.evaluate(z, sites)
        observedCovariance = covFcn.evaluate_covariance(sites, sites)
        observedCovariance = observedCovariance.to_dense() if isinstance(
            observedCovariance, DenseCovarianceMatrix
        ) else observedCovariance
        observedCovariance = backend_constant(observedCovariance, z)
        observedCovariance = observedCovariance + backend_constant(
            1e-8 * np.eye(observedCovariance.shape[-1]), z
        )
        kStar = covFcn.evaluate_covariance(queryGrid, sites)
        kStarArr = kStar.to_dense() if isinstance(
            kStar, DenseCovarianceMatrix) else kStar
        kStarArr = backend_constant(kStarArr, z)
        return kStarArr @ backend.solve(observedCovariance, observed)

    def create_predictor(
            self, gpState, queryGrid: Grid,
            coefficient: np.ndarray, observationGrid=None) -> Predictor:
        if observationGrid is not None:
            mean = self.evaluate_exact_conditional(
                gpState.expansion, queryGrid, coefficient,
                gpState.covarianceFunction, observationGrid,
            )
            return DirectGPPredictor(mean)
        shapeCovariance = gpState.expansion.shapeCovariance
        if not isinstance(shapeCovariance, DenseCovarianceMatrix):
            raise NotImplementedError(
                "Only DenseCovarianceMatrix is supported."
            )

        kStar = gpState.covarianceFunction.evaluate_covariance(
            queryGrid, self._grid)
        kStarArray = kStar.to_dense() if isinstance(
            kStar, DenseCovarianceMatrix) else kStar
        backend = infer_backend(coefficient)
        kStarArray = backend_constant(kStarArray, coefficient)
        factor = backend_constant(shapeCovariance.to_cholesky(), coefficient)
        mean = kStarArray @ backend.solve_triangular(
            factor.T, coefficient, lower=False
        )
        return DirectGPPredictor(mean)


class DirectGPPredictor(Predictor):
    """
    Immutable out-of-sample mean snapshot for the direct GP parametrisation.

    Parameters
    ----------
    mean : np.ndarray
        Predictive mean computed from the coefficient and covariance state
        at construction time.
    """

    def __init__(self, mean):
        self._mean = mean

    def mean(self):
        return self._mean
