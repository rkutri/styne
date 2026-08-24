import pytest
import numpy as np

from styne.gp.gaussianprocess import GaussianProcess
from styne.model.sglmm import SGLMM
from styne.parameter.vector import Vector
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid
from styne.model.trend import ConstantTrend

def test_sglmm_predictor_regression():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.direct(grid, covFcn)
    
    sglmm = SGLMM(gp, grid)
    queryGrid = UniformGrid(0., 1., 20)
    
    z = np.random.randn(gp.parameterDimension)
    predictor = sglmm.create_predictor(sglmm.prepare(Vector(z)), queryGrid)
    
    pred_mean = predictor.mean()
    
    realisation = gp.engine.build_realisation()
    realisation.coefficient = z
    exact_conditional = gp.engine.evaluate_exact_conditional(queryGrid, realisation, gp.covarianceFunction)
    
    assert np.allclose(pred_mean, exact_conditional, atol=1e-10)

def test_sglmm_predictor_regression_with_trend():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.direct(grid, covFcn)
    
    sglmm = SGLMM(gp, grid, trend=ConstantTrend(5.0))
    queryGrid = UniformGrid(0., 1., 20)
    
    z = np.random.randn(gp.parameterDimension)
    predictor = sglmm.create_predictor(sglmm.prepare(Vector(z)), queryGrid)
    
    pred_mean = predictor.mean()
    
    realisation = gp.engine.build_realisation()
    realisation.coefficient = z
    exact_conditional = gp.engine.evaluate_exact_conditional(queryGrid, realisation, gp.covarianceFunction)
    
    assert np.allclose(pred_mean, exact_conditional + 5.0, atol=1e-10)
