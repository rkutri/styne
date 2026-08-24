import numpy as np
from numpy.random import default_rng
from scipy.linalg import solve_triangular

from styne.gp.gaussianprocess import GaussianProcess
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid


def _setup(seed=42):
    rng = default_rng(seed)
    grid = UniformGrid(0.0, 1.0, 30)
    covariance = MaternCovariance1D(0.2, 1.5, 1.0)
    process = GaussianProcess.direct(grid, covariance)
    queryGrid = UniformGrid(0.0, 1.0, 10)
    coefficient = rng.standard_normal(process.parameterDimension)
    return process, queryGrid, coefficient


def test_direct_evaluation_matches_021_covariance_projection():
    process, queryGrid, coefficient = _setup()

    result = process.evaluate(coefficient, queryGrid)

    crossCovariance = process.covarianceFunction.evaluate_covariance(
        queryGrid, process.nativeGrid
    )
    factor = process.expansion.shapeCovariance.to_cholesky()
    expected = np.asarray(crossCovariance) @ solve_triangular(
        factor.T, coefficient, lower=False
    )
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_direct_bound_evaluation_caches_covariance_basis():
    process, queryGrid, coefficient = _setup()

    first = process.bind(queryGrid)
    firstResult = first.evaluate(coefficient)
    second = process.bind(queryGrid)

    assert second is first
    assert second._basis.shape == (
        len(queryGrid), process.parameterDimension
    )
    np.testing.assert_array_equal(
        second.evaluate(coefficient), firstResult
    )


def test_direct_evaluation_uses_explicit_coordinates():
    process, queryGrid, coefficient = _setup()
    other = default_rng(7).standard_normal(process.parameterDimension)

    first = process.evaluate(coefficient, queryGrid)
    second = process.evaluate(other, queryGrid)

    assert not np.allclose(first, second)
