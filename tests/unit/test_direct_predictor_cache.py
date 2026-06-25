import numpy as np
import pytest
from numpy.random import default_rng
from scipy.linalg import solve_triangular

from styne.gp.gaussianprocess import GaussianProcess
from styne.gp.direct import DirectGPPredictor
from styne.model.sglmm import SGLMM
from styne.statistics.stationary import (
    MaternCovariance1D, matern_covariance
)
from styne.utility.grid import UniformGrid


def _setup(seed=42):
    rng = default_rng(seed)
    grid = UniformGrid(0., 1., 30)
    covFcn = MaternCovariance1D(0.2, 1.5, 1.0)
    gp = GaussianProcess.direct(grid, covFcn)
    queryGrid = UniformGrid(0., 1., 10)
    predictor = gp.engine.create_predictor(gp, queryGrid)
    z = rng.standard_normal(gp.parameterDimension)
    gp.parameter.coordinate = z
    return gp, predictor, queryGrid, z


def test_mean_matches_explicit_computation():
    """Verify cached predictor gives same result as manual computation."""
    gp, predictor, queryGrid, z = _setup()

    pred = predictor.mean()

    covFcn = gp.covarianceFunction
    kStar = covFcn.evaluate_covariance(queryGrid, gp.engine.grid)
    kStarArr = np.asarray(kStar)
    L = gp.engine._shapeCovariance._cholFactor
    expected = kStarArr @ solve_triangular(L.T, z, lower=False)

    np.testing.assert_allclose(pred, expected, atol=1e-12)


def test_mean_reflects_state_update():
    """Changing the GP parameter must produce different predictions."""
    gp, predictor, queryGrid, z = _setup()
    rng = default_rng(99)

    pred1 = predictor.mean()
    gp.parameter.coordinate = rng.standard_normal(
        gp.parameterDimension
    )
    pred2 = predictor.mean()

    assert not np.allclose(pred1, pred2)


def test_mean_reflects_covariance_update():
    """Changing the covariance function must produce different predictions."""
    gp, predictor, queryGrid, z = _setup()
    pred1 = predictor.mean()

    gp.covarianceFunction = MaternCovariance1D(0.5, 1.5, 2.0)
    predictor2 = gp.engine.create_predictor(gp, queryGrid)
    pred2 = predictor2.mean()

    assert not np.allclose(pred1, pred2)
