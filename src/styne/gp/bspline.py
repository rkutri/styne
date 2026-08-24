import numpy as np

from numpy.linalg import LinAlgError

from styne.gp.engine import GPEngine, GPState
from styne.model.representation.bspline import BSpline2D
from styne.statistics.interface import CovarianceFunctionInterface, Predictor
from styne.statistics.covariance import CovarianceMatrix, DenseCovarianceMatrix
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

    y = np.linalg.solve(phi, kernel)
    c = np.linalg.solve(phi, y.T)
    c = 0.5 * (c + c.T)

    eps = nuggetEps
    for k in range(maxTries):

        try:
            return DenseCovarianceMatrix(c)

        except (ValueError, LinAlgError) as e:

            if k == maxTries - 1:
                raise ValueError(
                    "Prior covariance is not s.p.d. even after attempted "
                    "regularisation."
                ) from e

            c = c + eps * np.eye(n)
            eps *= 100.


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

    y = np.linalg.solve(phi2D, K)
    c = np.linalg.solve(phi2D, y.T)
    c = 0.5 * (c + c.T)

    eps = nuggetEps
    for k in range(maxTries):

        try:
            return DenseCovarianceMatrix(c)

        except (ValueError, LinAlgError) as e:

            if k == maxTries - 1:
                raise ValueError(
                    "2D prior covariance is not s.p.d. even after attempted "
                    "regularisation."
                ) from e

            c = c + eps * np.eye(N)
            eps *= 100.


class BSplineGPEngine(GPEngine):
    """
    GPEngine using a B-spline basis parametrisation, 1D or 2D depending on
    the expansion supplied.

    Parameters
    ----------
    expansion : BSpline1D | BSpline2D
        The B-spline basis. Dimensionality is inferred from its type.
    """

    def __init__(self, expansion):
        self._expansion = expansion
        self._is2d = isinstance(expansion, BSpline2D)
        self._H = None

    @property
    def spatialDimension(self) -> int:
        return 2 if self._is2d else 1

    def set_sites(self, sites: Grid) -> None:
        if sites is None:
            self._H = None
            return
        pts = sites.to_array()
        self._H = self._expansion.design_matrix(pts if self._is2d else pts.ravel())

    def build_expansion(self):
        """Return the engine's static B-spline representation."""
        return self._expansion

    def build_covariance(
            self, covFcn: CovarianceFunctionInterface) -> CovarianceMatrix:
        if self._is2d:
            bspX = self._expansion.splineX
            bspY = self._expansion.splineY
            return induced_prior_covariance_2d(
                covFcn, bspX, bspY, bspX.boundary, bspY.boundary
            )
        bounds = np.array(self._expansion.boundary)
        return induced_prior_covariance(covFcn, bounds, self._expansion)

    def apply_jacobian(
            self, vector: np.ndarray,
            covariance: CovarianceMatrix) -> np.ndarray:
        return vector @ self._H.T

    def apply_adjoint_jacobian(
            self, cotangent: np.ndarray,
            covariance: CovarianceMatrix) -> np.ndarray:
        return cotangent @ self._H

    def create_predictor(
            self, gpState: GPState, queryGrid: Grid,
            coefficient: np.ndarray) -> Predictor:
        pts = queryGrid.to_array()
        H_pred = self._expansion.design_matrix(pts if self._is2d else pts.ravel())
        frozenCoefficient = np.array(coefficient, dtype=float, copy=True)
        mean = H_pred @ frozenCoefficient if frozenCoefficient.ndim == 1 \
            else frozenCoefficient @ H_pred.T
        return BSplineGPPredictor(mean)


class BSplineGPPredictor(Predictor):
    """
    Immutable out-of-sample mean snapshot for the B-spline GP engine.

    Parameters
    ----------
    mean : np.ndarray
        Predictive mean computed from the coefficient at construction time.
    """

    def __init__(self, mean: np.ndarray):
        self._mean = np.array(mean, dtype=float, copy=True)

    def mean(self) -> np.ndarray:
        """
        Predictive mean at the query sites.

        Returns
        -------
        np.ndarray
            Predictive mean values, `H_pred @ coefficients`.
        """
        return self._mean.copy()
