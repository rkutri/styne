import numpy as np

from scipy.interpolate import make_interp_spline
from scipy.sparse import csr_matrix

from styne.model.representation.expansion import GridFunctionInterface
from styne.utility.grid import Grid


class Interpolation1D(GridFunctionInterface):
    """
    1D grid-function interpolation via `scipy.interpolate.make_interp_spline`.

    Natural boundary conditions are used when `degree >= 3`, unconstrained
    otherwise.

    Parameters
    ----------
    grid : array_like
        1D grid coordinates.
    value : array_like
        Values at `grid`.
    degree : int, default 3
        Spline degree.
    """

    def __init__(self, grid, value, degree=3):

        bcType = 'natural' if degree >= 3 else None
        self._interp = make_interp_spline(grid, value, degree, bc_type=bcType)

    def evaluate(self, grid: Grid) -> np.ndarray:
        """
        Evaluate the spline at a query grid's points.

        Parameters
        ----------
        grid : Grid

        Returns
        -------
        np.ndarray
        """
        return self._interp(grid.to_array().ravel())


class GridInterpolation2D(GridFunctionInterface):
    """
    Bilinear interpolant on a regular 2D grid.

    Parameters
    ----------
    gridX : array_like
        Strictly increasing x-coordinates.
    gridY : array_like
        Strictly increasing y-coordinates.
    values : array_like, shape (len(gridX), len(gridY))
        Grid values. Out-of-bounds queries extrapolate rather than raising,
        `bounds_error=False` with no fixed fill value.
    """

    def __init__(self, gridX, gridY, values):

        from scipy.interpolate import RegularGridInterpolator

        self._interp = RegularGridInterpolator(
            (gridX, gridY), values, method='linear',
            bounds_error=False, fill_value=None
        )

    def evaluate(self, grid: Grid) -> np.ndarray:
        """
        Evaluate the bilinear interpolant at a query grid's points.

        Parameters
        ----------
        grid : Grid

        Returns
        -------
        np.ndarray
        """
        return self._interp(grid.to_array())


def linear_interpolation_matrix(queryPoints, grid):
    """
    Sparse linear interpolation matrix $I \in \mathbb{R}^{N \times n_{grid}}$
    (CSR format).

    Maps values on a 1D grid to N arbitrary query points via linear
    interpolation. Node ordering, $\text{node}(i) = i$, consistent with
    `evaluate_native()` for $d=1$.

    Parameters
    ----------
    queryPoints : ndarray, shape (N,) or (N, 1)
        Query coordinates.
    grid : ndarray, shape (nGrid,)
        Strictly increasing 1D grid nodes.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (N, nGrid)
    """
    import warnings

    queryPoints = np.asarray(queryPoints).ravel()
    grid = np.asarray(grid)
    nPts = len(queryPoints)
    nGrid = len(grid)

    if np.any((queryPoints < grid[0]) | (queryPoints > grid[-1])):
        warnings.warn(
            "Some query points lie outside the grid boundaries. "
            "Linear extrapolation will be performed.",
            UserWarning
        )

    ix = np.searchsorted(grid, queryPoints, side='right') - 1
    ix = np.clip(ix, 0, nGrid - 2)

    tx = (queryPoints - grid[ix]) / (grid[ix + 1] - grid[ix])

    rows = np.repeat(np.arange(nPts), 2)
    cols = np.column_stack((ix, ix + 1)).ravel()
    vals = np.column_stack((1.0 - tx, tx)).ravel()

    return csr_matrix(
        (vals, (rows, cols)),
        shape=(nPts, nGrid)
    )


def bilinear_interpolation_matrix(queryPoints, gridX, gridY):
    """
    Sparse bilinear interpolation matrix
    $I \in \mathbb{R}^{N \times n_X n_Y}$ (CSR format).

    Maps values on a regular 2D grid to N arbitrary query points via
    bilinear interpolation. Node ordering, $\text{node}(i,j) = i n_Y + j$,
    consistent with the flat row-major layout of
    `DNAFourierRealisation.evaluate_native()`.

    Parameters
    ----------
    queryPoints : ndarray, shape (N, 2)
        Query coordinates, each row is `(x, y)`.
    gridX : ndarray, shape (nX,)
        Strictly increasing x-coordinates of the grid.
    gridY : ndarray, shape (nY,)
        Strictly increasing y-coordinates of the grid.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (N, nX*nY)
    """
    import warnings

    queryPoints = np.asarray(queryPoints)
    gridX = np.asarray(gridX)
    gridY = np.asarray(gridY)

    nPts = queryPoints.shape[0]
    nX = len(gridX)
    nY = len(gridY)

    px = queryPoints[:, 0]
    py = queryPoints[:, 1]

    outX = (px < gridX[0]) | (px > gridX[-1])
    outY = (py < gridY[0]) | (py > gridY[-1])
    if np.any(outX | outY):
        warnings.warn(
            "Some query points lie outside the grid boundaries. "
            "Bilinear extrapolation will be performed.",
            UserWarning
        )

    ix = np.searchsorted(gridX, px, side='right') - 1
    iy = np.searchsorted(gridY, py, side='right') - 1

    ix = np.clip(ix, 0, nX - 2)
    iy = np.clip(iy, 0, nY - 2)

    tx = (px - gridX[ix]) / (gridX[ix + 1] - gridX[ix])
    ty = (py - gridY[iy]) / (gridY[iy + 1] - gridY[iy])

    rows = np.repeat(np.arange(nPts), 4)

    c00 = ix * nY + iy
    c01 = ix * nY + iy + 1
    c10 = (ix + 1) * nY + iy
    c11 = (ix + 1) * nY + iy + 1
    cols = np.column_stack((c00, c01, c10, c11)).ravel()

    v00 = (1.0 - tx) * (1.0 - ty)
    v01 = (1.0 - tx) * ty
    v10 = tx * (1.0 - ty)
    v11 = tx * ty
    vals = np.column_stack((v00, v01, v10, v11)).ravel()

    return csr_matrix(
        (vals, (rows, cols)),
        shape=(nPts, nX * nY)
    )
