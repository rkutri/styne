import numpy as np
from numpy.random import default_rng
from scipy.linalg import solve_triangular

from styne.gp.gaussianprocess import GaussianProcess
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid


def _setup(seed=42):
    rng = default_rng(seed)
    grid = UniformGrid(0., 1., 30)
    covFcn = MaternCovariance1D(0.2, 1.5, 1.0)
    gp = GaussianProcess.direct(grid, covFcn)
    queryGrid = UniformGrid(0., 1., 10)
    z = rng.standard_normal(gp.parameterDimension)
    return gp, queryGrid, z


def test_mean_matches_explicit_computation():
    gp, queryGrid, z = _setup()
    predictor = gp.engine.create_predictor(gp, queryGrid, z)
    pred = predictor.mean()

    covFcn = gp.covarianceFunction
    kStar = covFcn.evaluate_covariance(queryGrid, gp.engine.grid)
    kStarArr = np.asarray(kStar)
    L = gp.engine._shapeCovariance._cholFactor
    expected = kStarArr @ solve_triangular(L.T, z, lower=False)

    np.testing.assert_allclose(pred, expected, atol=1e-12)


def test_predictor_is_a_snapshot():
    gp, queryGrid, z = _setup()
    predictor = gp.engine.create_predictor(gp, queryGrid, z)
    pred1 = predictor.mean()

    z[:] = default_rng(7).standard_normal(gp.parameterDimension)
    gp.parameter.coordinate = default_rng(99).standard_normal(
        gp.parameterDimension)
    pred2 = predictor.mean()

    np.testing.assert_array_equal(pred1, pred2)

    pred2[:] = 0.0
    np.testing.assert_array_equal(pred1, predictor.mean())


def test_distinct_coefficients_give_distinct_predictors():
    gp, queryGrid, z = _setup()
    other = default_rng(7).standard_normal(gp.parameterDimension)

    first = gp.engine.create_predictor(gp, queryGrid, z)
    second = gp.engine.create_predictor(gp, queryGrid, other)

    assert not np.allclose(first.mean(), second.mean())


def test_mean_reflects_covariance_update():
    gp, queryGrid, z = _setup()
    pred1 = gp.engine.create_predictor(gp, queryGrid, z).mean()

    gp.covarianceFunction = MaternCovariance1D(0.5, 1.5, 2.0)
    pred2 = gp.engine.create_predictor(gp, queryGrid, z).mean()

    assert not np.allclose(pred1, pred2)


def test_existing_predictor_ignores_covariance_update():
    gp, queryGrid, z = _setup()
    predictor = gp.engine.create_predictor(gp, queryGrid, z)
    expected = predictor.mean()

    gp.covarianceFunction = MaternCovariance1D(0.5, 1.5, 2.0)

    np.testing.assert_array_equal(expected, predictor.mean())
