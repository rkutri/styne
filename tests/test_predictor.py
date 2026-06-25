import pytest
import numpy as np

from styne.gp.gaussianprocess import GaussianProcess
from styne.model.sglmm import SGLMM
from styne.parameter.block import BlockParameter
from styne.parameter.vector import Vector
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid


def test_dna_predictor_equivalence():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.dna(covFcn, q=10, d=1)
    
    queryGrid = UniformGrid(0., 1., 20)
    predictor = gp._engine.create_predictor(gp, queryGrid)
    
    gp.parameter.coordinate = np.random.randn(gp.parameterDimension)
    pred1 = predictor.mean()
    gp.sites = queryGrid
    pred2 = gp.at_sites()
    
    assert np.allclose(pred1, pred2)

def test_dna_predictor_batching():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.dna(covFcn, q=10, d=1)
    
    queryGrid = UniformGrid(0., 1., 20)
    predictor = gp._engine.create_predictor(gp, queryGrid)
    
    nBatch = 5
    gp.parameter.coordinate = np.random.randn(nBatch, gp.parameterDimension)
    pred_batch = predictor.mean()
    
    assert pred_batch.shape == (nBatch, 20)

def test_sglmm_predictor_features():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.direct(grid, covFcn)
    
    X = np.random.randn(100, 2)
    sglmm = SGLMM(gp, grid, features=X)
    
    queryGrid = UniformGrid(0., 1., 20)
    
    with pytest.raises(ValueError, match="Out-of-sample features required for prediction."):
        sglmm.create_predictor(queryGrid)

def test_sglmm_predictor_shape():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.direct(grid, covFcn)
    
    X = np.random.randn(100, 2)
    sglmm = SGLMM(gp, grid, features=X)
    
    queryGrid = UniformGrid(0., 1., 20)
    X_pred_wrong = np.random.randn(10, 2)
    
    with pytest.raises(ValueError, match="features must have shape"):
        sglmm.create_predictor(queryGrid, features=X_pred_wrong)

def test_sglmm_predictor_missing_trend():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.direct(grid, covFcn)
    
    sglmm = SGLMM(gp, grid)
    
    queryGrid = UniformGrid(0., 1., 20)
    predictor = sglmm.create_predictor(queryGrid)
    
    gp.parameter.coordinate = np.zeros(gp.parameterDimension)
    val = predictor.mean()
    assert val is not None

def test_sglmm_caching_and_fixed_effects():
    """
    Ensures that fixed effects are evaluated correctly out-of-sample,
    and formalises the `reset()` model cache pattern to prevent regressions.
    """
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.direct(grid, covFcn)
    
    X_obs = np.random.randn(100, 2)
    sglmm = SGLMM(gp, grid, features=X_obs)
    
    queryGrid = UniformGrid(0., 1., 20)
    X_pred = np.random.randn(20, 2)
    
    predictor = sglmm.create_predictor(queryGrid, features=X_pred)
    
    latent = Vector(np.random.randn(gp.parameterDimension))
    fixed = Vector(np.array([1.0, -0.5]))
    param = BlockParameter([latent, fixed])
    
    sglmm.reset()
    sglmm.interpolate(param)
    
    mean1 = predictor.mean()
    
    # Mutate in-place, simulating MCMC behaviour
    param.block(1).coordinate = np.array([2.0, 1.0])
    
    # Interpolating WITHOUT reset will hit the cache return
    sglmm.interpolate(param)
    mean_stale = predictor.mean()
    assert np.allclose(mean1, mean_stale), "Expected stale cache if reset() is not called"
    
    # Now explicitly reset and interpolate
    sglmm.reset()
    sglmm.interpolate(param)
    mean_fresh = predictor.mean()
    
    assert not np.allclose(mean1, mean_fresh), "Expected update after explicit reset()"
