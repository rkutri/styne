import numpy as np

from styne.backend import infer_backend
from styne.model.representation.expansion import backend_constant

from styne.model.representation.bspline import BSpline2D
from styne.statistics.interface import CovarianceFunctionInterface, Predictor
from styne.statistics.covariance import DenseCovarianceMatrix
from styne.utility.grid import Grid


def induced_prior_covariance(
        cov_fn, grid, expansion, nuggetEps=1e-10, maxTries=3):
    """
    Approximation of the GP prior over the finite-dimensional space spanned by
    the expansion. Not a projection; the exact GP covariance is evaluated only
    at a finite set of collocation points.

    Parameters
    ----------
    cov_fn : object with evaluate_covariance(pts1, pts2)
        Stationary covariance function (e.g. MaternCovariance1D).
    grid : array_like, shape (n,)
        1D GP grid; determines the domain [grid[0], grid[-1]].
    expansion : Expansion
        Finite-dimensional basis (e.g. BSpline1D).
    """

    grid = np.asarray(grid)
    if grid.ndim != 1:
        raise ValueError("Only 1D GP grids are supported for now.")

    n = expansion.dimension

    x0, x1 = grid[0], grid[-1]
    collocation = np.linspace(x0, x1, n)

    phi = expansion.design_matrix(collocation)
    kernel = cov_fn.evaluate_covariance(collocation, collocation)
    backend = infer_backend(kernel)
    phi = backend_constant(phi, kernel)

    y = backend.solve(phi, kernel)
    c = backend.solve(phi, y.T)
    c = 0.5 * (c + c.T)
    return DenseCovarianceMatrix(
        c + nuggetEps * backend.eye(
            n, dtype=backend.metadata(c).dtype, device=backend.metadata(c).device
        )
    )


def induced_prior_covariance_2d(covFunc2d, bspX, bspY, xBounds, yBounds,
                                nuggetEps=1e-10, maxTries=3):
    """
    Induced prior covariance in 2D tensor-product B-spline coefficient space
    for an *isotropic* 2D covariance kernel.

    Parameters
    ----------
    covFunc2d : object with evaluate_covariance(pts1, pts2) -> (N,M) ndarray
        2D covariance function, e.g. MaternCovariance2D.
    bspX, bspY : BSpline1D
        The two marginal B-spline bases (must each have design_matrix()).
    xBounds, yBounds : [float, float]
        Domain bounds [a, b] for each axis.
    """

    nx = bspX.dimension
    ny = bspY.dimension
    N = nx * ny

    xColloc = np.linspace(xBounds[0], xBounds[1], nx)
    yColloc = np.linspace(yBounds[0], yBounds[1], ny)

    xi, yj = np.meshgrid(xColloc, yColloc, indexing='ij')
    pts = np.column_stack([xi.ravel(), yj.ravel()])

    K = covFunc2d.evaluate_covariance(pts, pts)

    phiX = bspX.design_matrix(xColloc)
    phiY = bspY.design_matrix(yColloc)
    phi2D = np.kron(phiX, phiY)

    backend = infer_backend(K)
    phi2D = backend_constant(phi2D, K)
    y = backend.solve(phi2D, K)
    c = backend.solve(phi2D, y.T)
    c = 0.5 * (c + c.T)
    return DenseCovarianceMatrix(
        c + nuggetEps * backend.eye(
            N, dtype=backend.metadata(c).dtype, device=backend.metadata(c).device
        )
    )


class _BSplineGPSpecification:
    """Construction and prediction rules for a B-spline GP."""

    def __init__(self, expansion):
        self._expansion = expansion
        self._is2d = isinstance(expansion, BSpline2D)

    @property
    def spatialDimension(self) -> int:
        return 2 if self._is2d else 1

    def build(self, covFcn: CovarianceFunctionInterface):
        if self._is2d:
            bspX = self._expansion.splineX
            bspY = self._expansion.splineY
            covariance = induced_prior_covariance_2d(
                covFcn, bspX, bspY, bspX.boundary, bspY.boundary
            )
        else:
            bounds = np.array(self._expansion.boundary)
            covariance = induced_prior_covariance(
                covFcn, bounds, self._expansion
            )
        return covariance, self._expansion

    def create_predictor(
            self, gpState, queryGrid: Grid,
            coefficient: np.ndarray, observationGrid=None) -> Predictor:
        mean = self._expansion.evaluate(coefficient, queryGrid)
        return BSplineGPPredictor(mean)


class BSplineGPPredictor(Predictor):
    """
    Immutable out-of-sample mean snapshot for the B-spline parametrisation.

    Parameters
    ----------
    mean : np.ndarray
        Predictive mean computed from the coefficient at construction time.
    """

    def __init__(self, mean):
        self._mean = mean

    def mean(self):
        """
        Predictive mean at the query sites.

        Returns
        -------
        np.ndarray
            Predictive mean values, `H_pred @ coefficients`.
        """
        return self._mean
