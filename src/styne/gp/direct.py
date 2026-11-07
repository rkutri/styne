import numpy as np
from styne.backend import infer_backend
from styne.model.representation.expansion import (
    BoundLinearExpansion,
    LinearExpansion,
    backend_constant,
)
from styne.statistics.interface import CovarianceFunctionInterface
from styne.statistics.covariance import CovarianceMatrix, DenseCovarianceMatrix, IIDCovarianceMatrix
from styne.utility.grid import Grid
from styne.utility.interpolation import linear_interpolation_matrix


class DirectEvaluation(BoundLinearExpansion):

    def __init__(self, basis):
        self._basis = basis

    @property
    def dimension(self):
        return self._basis.shape[1]

    def evaluate(self, coefficient):
        if coefficient.ndim < 1 or coefficient.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected coefficient shape (..., {self.dimension}); "
                f"got {coefficient.shape}."
            )
        basis = backend_constant(self._basis, coefficient)
        return coefficient @ basis.T

    def _adjoint_derivative(self, coefficient, cotangent):
        basis = backend_constant(self._basis, cotangent)
        return cotangent @ basis


class DirectExpansion(LinearExpansion):
    """
    A Gaussian-process parametrisation on a fixed representation grid.

    When ``shapeCovariance`` is present, coordinates are whitened and its
    Cholesky factor maps them to field values on the representation grid.
    A supplied ``covarianceFunction`` extends those values off-grid with the
    covariance basis ``K(query, grid) @ L**-T``. Without one, the expansion
    retains ordinary piecewise-linear interpolation in one dimension.

    Parameters
    ----------
    grid : Grid
        The grid on which the direct coefficients live.
    dimension : int
        Number of coefficients (grid points).
    """

    def __init__(
            self, grid: Grid, dimension: int,
            shapeCovariance: CovarianceMatrix = None,
            covarianceFunction: CovarianceFunctionInterface = None):

        self._dim = dimension
        self._grid = grid
        self._shapeCov = shapeCovariance
        self._covFcn = covarianceFunction

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
        if self._shapeCov is not None and self._covFcn is not None:
            factor = self._shapeCov.to_cholesky()
            if np.array_equal(query, anchor):
                basis = factor
            else:
                crossCovariance = self._covFcn.evaluate_covariance(
                    grid, self._grid
                )
                if isinstance(crossCovariance, DenseCovarianceMatrix):
                    crossCovariance = crossCovariance.to_dense()
                crossCovariance = backend_constant(
                    crossCovariance, factor
                )
                backend = infer_backend(factor)
                basis = backend.solve_triangular(
                    factor, crossCovariance.T, lower=True
                ).T
            return DirectEvaluation(basis)

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

        basis = interpolation
        if self._shapeCov is not None:
            factor = self._shapeCov.to_cholesky()
            basis = backend_constant(interpolation, factor) @ factor
        return DirectEvaluation(basis)


class DirectGPSpecification:
    """Construction rules for a direct GP parametrisation."""

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
            self._grid, len(self._grid), shapeCovariance,
            covarianceFunction
        )
        unitVariance = shapeCovariance.scaling * 0.0 + 1.0
        return IIDCovarianceMatrix(
            len(self._grid), unitVariance
        ), expansion

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
