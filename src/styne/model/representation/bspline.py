import numpy as np
import scipy.interpolate as si

from typing import List

from styne.model.representation.expansion import (
    BoundLinearExpansion,
    LinearExpansion,
    backend_constant,
)
from styne.utility.grid import Grid


def _validate_coefficient(coefficient, dimension):
    if coefficient.ndim < 1:
        raise ValueError(
            "coefficient must have shape (..., dimension)"
        )
    if coefficient.shape[-1] != dimension:
        raise ValueError(
            f"Expected {dimension} coefficients, "
            f"got {coefficient.shape[-1]}."
        )
    return coefficient


def _grid_array(grid):
    return grid.to_array() if isinstance(grid, Grid) else np.asarray(grid)


class _BSplineEvaluation(BoundLinearExpansion):

    def __init__(self, designMatrix):
        self._designMatrix = np.asarray(designMatrix)

    @property
    def dimension(self):
        return self._designMatrix.shape[1]

    def evaluate(self, coefficient):
        coefficient = _validate_coefficient(coefficient, self.dimension)
        designMatrix = backend_constant(self._designMatrix, coefficient)
        return coefficient @ designMatrix.T

    def _adjoint_derivative(self, coefficient, cotangent):
        return cotangent @ backend_constant(self._designMatrix, cotangent)


class BSpline2D(LinearExpansion):
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

    def _bind(self, grid) -> BoundLinearExpansion:
        return _BSplineEvaluation(self.design_matrix(grid))


class BSpline1D(LinearExpansion):

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

    def _bind(self, grid) -> BoundLinearExpansion:
        return _BSplineEvaluation(self.design_matrix(grid))
