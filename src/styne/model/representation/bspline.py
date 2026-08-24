import numpy as np
import scipy.interpolate as si

from typing import List

from styne.model.representation.expansion import Expansion
from styne.utility.grid import Grid


def _validate_coefficient(coefficient):
    coefficient = np.asarray(coefficient, dtype=float)
    if coefficient.ndim < 1:
        raise ValueError(
            "coefficient must have shape (..., dimension)"
        )
    return coefficient


def _evaluate_design_matrix(designMatrix, coefficient, dimension):
    coefficient = _validate_coefficient(coefficient)
    if coefficient.shape[-1] != dimension:
        raise ValueError(
            f"Expected {dimension} coefficients, "
            f"got {coefficient.shape[-1]}."
        )
    return coefficient @ designMatrix.T


def _grid_array(grid):
    return grid.to_array() if isinstance(grid, Grid) else np.asarray(grid)


class BSpline2D(Expansion):
    """
    Tensor-product B-spline on a 2D rectangular domain.

    The coefficient vector is flat with length nx*ny, ordered row-major:
    coeff[i*ny + j] multiplies B_i(x) * B_j(y).
    """

    def __init__(self,
                 nBasis: List[int],
                 degree: int = 3,
                 boundary: List[List[float]] = [[0., 1.], [0., 1.]]
                 ):

        if len(nBasis) != 2:
            raise ValueError("nBasis must contain exactly two entries.")

        if len(boundary) != 2 or any(len(b) != 2 for b in boundary):
            raise ValueError("boundary must be [[x0, x1], [y0, y1]].")

        self._nBasis = list(nBasis)
        self._splineX = BSpline1D(nBasis[0], degree, boundary[0])
        self._splineY = BSpline1D(nBasis[1], degree, boundary[1])

    @property
    def dimension(self) -> int:
        return self._nBasis[0] * self._nBasis[1]

    @property
    def splineX(self) -> 'BSpline1D':
        return self._splineX

    @property
    def splineY(self) -> 'BSpline1D':
        return self._splineY

    def design_matrix(self, grid: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        grid : ndarray of shape (N, 2)
            2D evaluation points.

        Returns
        -------
        ndarray of shape (N, nx*ny)
            Tensor-product design matrix. Entry [k, i*ny+j] equals
            phi_i(grid[k,0]) * psi_j(grid[k,1]).
        """

        grid = _grid_array(grid)
        N = grid.shape[0]
        nx, ny = self._nBasis

        PhiX = self._splineX.design_matrix(grid[:, 0])   # (N, nx)
        PhiY = self._splineY.design_matrix(grid[:, 1])   # (N, ny)

        # result[k, i*ny+j] = PhiX[k,i] * PhiY[k,j]
        return (PhiX[:, :, None] * PhiY[:, None, :]).reshape(N, nx * ny)

    def evaluate(self, coefficient, grid: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        grid : ndarray of shape (N, 2)

        Returns
        -------
        ndarray of shape (N,)
        """
        return _evaluate_design_matrix(
            self.design_matrix(grid), coefficient, self.dimension
        )


class BSpline1D(Expansion):

    def __init__(self,
                 nBasis: int,
                 degree: int = 3,
                 boundary: List[float] = [0., 1.]
                 ):

        if len(boundary) != 2:
            raise ValueError("1D boundary must consist of two points.")

        if nBasis <= degree:
            raise ValueError(
                f"B-Spline of degree {degree} requires at least a"
                f" {degree + 1}-dimensional coefficient vector."
                f" Got {nBasis}."
            )

        self._nBasis = nBasis
        self._boundary = boundary
        self._degree = degree

        self._knots = self._set_bspline_knots()

    @property
    def dimension(self) -> int:
        return self._nBasis

    @property
    def degree(self):
        return self._degree

    @property
    def boundary(self):
        return list(self._boundary)

    def design_matrix(self, grid: np.ndarray) -> np.ndarray:
        grid = _grid_array(grid).ravel()
        return si.BSpline.design_matrix(
            grid, self._knots, self.degree, extrapolate=False
        ).toarray()

    def spline_interval(self):
        return self._knots[
            self._degree:self._degree + self._nBasis
        ]

    def _set_bspline_knots(self) -> np.ndarray:
        """
        Evenly spaced interior knots and clamping at the boundary
        """

        leftClamp = np.full(self._degree, self._boundary[0])
        rightClamp = np.full(self._degree, self._boundary[1])

        nDom = self._nBasis - self._degree + 1
        domain = np.linspace(
            self._boundary[0], self._boundary[1], nDom, endpoint=True
        )

        return np.concatenate((leftClamp, domain, rightClamp))

    def evaluate(self, coefficient, grid: np.ndarray) -> np.ndarray:
        return _evaluate_design_matrix(
            self.design_matrix(grid), coefficient, self.dimension
        )
