import numpy as np
import pytest

from styne.utility.interpolation import (
    Interpolation1D, GridInterpolation2D,
    linear_interpolation_matrix, bilinear_interpolation_matrix
)
from styne.utility.grid import Grid


class TestInterpolation1D:

    def test_exact_at_nodes(self):
        grid = np.linspace(0., 1., 6)
        values = np.array([1., 3., 2., 5., 4., 2.])
        interp = Interpolation1D(grid, values, degree=1)
        result = interp.evaluate(Grid(grid[:, None]))
        np.testing.assert_allclose(result, values, atol=1e-12)

    def test_midpoint_linear(self):
        grid = np.array([0., 1., 2.])
        values = np.array([0., 4., 2.])
        interp = Interpolation1D(grid, values, degree=1)
        result = interp.evaluate(Grid(np.array([0.5, 1.5])[:, None]))
        expected = np.array([2., 3.])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_cubic_smooth(self):
        grid = np.linspace(0., 1., 6)
        values = np.array([1., 3., 2., 5., 4., 2.])
        interp = Interpolation1D(grid, values, degree=3)
        midpoints = 0.5 * (grid[:-1] + grid[1:])
        result = interp.evaluate(Grid(midpoints[:, None]))
        # cubic interpolant should not simply clamp to neighbourhood extremes —
        # just verify the call succeeds and returns the right shape
        assert result.shape == midpoints.shape


class TestGridInterpolation2D:

    def test_exact_at_nodes(self):
        gridX = np.array([0., 1., 2.])
        gridY = np.array([0., 1.])
        values = np.array([[1., 2.], [3., 4.], [5., 6.]])
        interp = GridInterpolation2D(gridX, gridY, values)
        pts = np.array([[x, y] for x in gridX for y in gridY])
        expected = np.array([values[i, j]
                             for i in range(len(gridX))
                             for j in range(len(gridY))])
        result = interp.evaluate(Grid(pts))
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_bilinear_midpoint(self):
        gridX = np.array([0., 1.])
        gridY = np.array([0., 1.])
        # f(x,y) = 1 + 2x + 3y + 4xy
        values = np.array([[1., 4.], [3., 10.]])
        interp = GridInterpolation2D(gridX, gridY, values)
        x, y = 0.5, 0.5
        expected = 1. + 2.*x + 3.*y + 4.*x*y
        result = interp.evaluate(Grid(np.array([[x, y]])))
        np.testing.assert_allclose(result[0], expected, atol=1e-12)


class TestInterpolationMatrices:

    def test_linear_interpolation_matrix_correctness(self):
        grid = np.array([0.0, 1.0, 2.0])
        query = np.array([0.5, 1.5])
        matrix = linear_interpolation_matrix(query, grid)
        values = np.array([10.0, 20.0, 15.0])
        result = matrix @ values
        expected = np.array([15.0, 17.5])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_linear_interpolation_matrix_extrapolation(self):
        grid = np.array([0.0, 1.0, 2.0])
        query = np.array([-0.5, 2.5])
        values = np.array([10.0, 20.0, 15.0])

        with pytest.warns(UserWarning, match="outside the grid boundaries"):
            matrix = linear_interpolation_matrix(query, grid)

        result = matrix @ values
        expected = np.array([5.0, 12.5])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_bilinear_interpolation_matrix_correctness(self):
        gridX = np.array([0.0, 1.0])
        gridY = np.array([0.0, 1.0])
        query = np.array([[0.5, 0.5]])
        values = np.array([1.0, 4.0, 3.0, 10.0])
        matrix = bilinear_interpolation_matrix(query, gridX, gridY)
        result = matrix @ values
        expected = np.array([4.5])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_bilinear_interpolation_matrix_extrapolation(self):
        gridX = np.array([0.0, 1.0])
        gridY = np.array([0.0, 1.0])
        query = np.array([[-0.5, 0.5], [1.5, 1.5]])
        values = np.array([1.0, 4.0, 3.0, 10.0])

        with pytest.warns(UserWarning, match="outside the grid boundaries"):
            matrix = bilinear_interpolation_matrix(query, gridX, gridY)

        result = matrix @ values
        expected = np.array([0.5, 17.5])
        np.testing.assert_allclose(result, expected, atol=1e-12)
