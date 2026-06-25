import numpy as np
import scipy.interpolate as si

from typing import List

from styne.model.representation.expansion import Expansion


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
        self._coeff = None

    @property
    def dimension(self) -> int:
        return self._nBasis[0] * self._nBasis[1]

    @property
    def splineX(self) -> 'BSpline1D':
        return self._splineX

    @property
    def splineY(self) -> 'BSpline1D':
        return self._splineY

    @property
    def coefficient(self) -> np.ndarray:

        if self._coeff is None:
            raise ValueError(
                "Trying to retrieve coefficient before BSpline2D setup."
            )

        return self._coeff.copy()

    @coefficient.setter
    def coefficient(self, coefficient: np.ndarray) -> None:
        self.project(self.validate(coefficient))

    def project(self, coefficient: np.ndarray) -> None:

        if coefficient.size != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} coefficients, "
                f"got {coefficient.size}."
            )

        self._coeff = coefficient

        # Initialise 1D splines with dummy coefficients to set up knots,
        # which are needed for design_matrix calls.
        self._splineX.coefficient = np.zeros(self._nBasis[0])
        self._splineY.coefficient = np.zeros(self._nBasis[1])

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

        if self._coeff is None:
            raise ValueError(
                "Trying to retrieve design matrix before BSpline2D is set up."
            )

        grid = np.asarray(grid)
        N = grid.shape[0]
        nx, ny = self._nBasis

        PhiX = self._splineX.design_matrix(grid[:, 0])   # (N, nx)
        PhiY = self._splineY.design_matrix(grid[:, 1])   # (N, ny)

        # result[k, i*ny+j] = PhiX[k,i] * PhiY[k,j]
        return (PhiX[:, :, None] * PhiY[:, None, :]).reshape(N, nx * ny)

    def evaluate(self, grid: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        grid : ndarray of shape (N, 2)

        Returns
        -------
        ndarray of shape (N,)
        """
        return self.design_matrix(grid) @ self._coeff


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

        self._bspline = None

    @property
    def dimension(self) -> int:

        if self._bspline is None:
            return self._nBasis

        if self._nBasis != self._bspline.c.size:
            raise ValueError("Value mismatch in BSpline coefficient size.")

        return self._nBasis

    @property
    def coefficient(self) -> np.ndarray:

        if self._bspline is None:
            raise ValueError(
                "Trying to retrieve coefficient before BSpline setup."
            )

        # return read-only coefficient (copy)
        return self._bspline.tck[1]

    @coefficient.setter
    def coefficient(self, coefficient: np.ndarray) -> None:
        self.project(self.validate(coefficient))

    @property
    def degree(self):
        return self._degree

    @property
    def boundary(self):
        return list(self._boundary)

    def design_matrix(self, grid: np.ndarray) -> np.ndarray:

        if self._bspline is None:
            raise ValueError(
                "Trying to retrieve design matrix before Spline is set up."
            )

        t = self._bspline.t
        k = self.degree

        return self._bspline.design_matrix(grid, t, k).toarray()

    def spline_interval(self):

        tck = self._bspline.tck
        return tck[0][tck[2]:tck[2] + tck[1].size]

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

    def project(self, coefficient: np.ndarray) -> None:

        t = self._set_bspline_knots()

        self._bspline = si.BSpline(
            t, coefficient, self._degree, extrapolate=False
        )

    def evaluate(self, grid: np.ndarray) -> np.ndarray:
        return self._bspline(grid)
