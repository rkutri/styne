import pytest
import numpy as np

from styne.gp.gaussianprocess import GaussianProcess
from styne.model.sglmm import SGLMM
from styne.model.representation.bspline import BSpline1D
from styne.parameter.block import BlockParameter
from styne.parameter.vector import Vector
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid


def test_dna_predictor_equivalence():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.dna(covFcn, q=10, d=1)
    
    queryGrid = UniformGrid(0., 1., 20)
    coefficient = np.random.randn(gp.parameterDimension)
    predictor = gp._engine.create_predictor(gp, queryGrid, coefficient)

    pred1 = predictor.mean()
    gp.sites = queryGrid
    pred2 = gp.at_sites(coefficient)
    
    assert np.allclose(pred1, pred2)

def test_dna_predictor_batching():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.dna(covFcn, q=10, d=1)
    
    queryGrid = UniformGrid(0., 1., 20)
    nBatch = 5
    coefficient = np.random.randn(nBatch, gp.parameterDimension)
    predictor = gp._engine.create_predictor(gp, queryGrid, coefficient)

    pred_batch = predictor.mean()
    
    assert pred_batch.shape == (nBatch, 20)


def test_dna_predictor_is_an_immutable_snapshot():
    gp = GaussianProcess.dna(
        MaternCovariance1D(1.0, 1.5, 1.5), q=10, d=1)
    coefficient = np.random.default_rng(1).standard_normal(
        gp.parameterDimension)
    predictor = gp.engine.create_predictor(
        gp, UniformGrid(0., 1., 20), coefficient)
    expected = predictor.mean()

    coefficient[:] = 0.0
    returned = predictor.mean()
    returned[:] = 0.0

    np.testing.assert_array_equal(expected, predictor.mean())


def test_bspline_predictor_is_an_immutable_snapshot():
    expansion = BSpline1D(6, degree=3, boundary=[0., 1.])
    expansion.project(np.zeros(6))
    gp = GaussianProcess.bspline(
        MaternCovariance1D(0.3, 1.5, 0.7), expansion)
    coefficient = np.random.default_rng(2).standard_normal(6)
    predictor = gp.engine.create_predictor(
        gp, UniformGrid(0., 1., 20), coefficient)
    expected = predictor.mean()

    coefficient[:] = 0.0
    returned = predictor.mean()
    returned[:] = 0.0

    np.testing.assert_array_equal(expected, predictor.mean())


def test_sglmm_create_predictor_does_not_mutate_gp():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.dna(covFcn, q=10, d=1)
    sglmm = SGLMM(gp, grid)

    before = np.array(gp.parameter.coordinate, copy=True)
    parameter = Vector(
        np.random.default_rng(3).standard_normal(gp.parameterDimension))
    sglmm.create_predictor(sglmm.prepare(parameter), UniformGrid(0., 1., 20))

    np.testing.assert_array_equal(gp.parameter.coordinate, before)

def test_sglmm_predictor_features():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.direct(grid, covFcn)
    
    X = np.random.randn(100, 2)
    sglmm = SGLMM(gp, grid, features=X)
    queryGrid = UniformGrid(0., 1., 20)
    latent = gp.parameter.clone()
    latent.coordinate = np.zeros(gp.parameterDimension)
    preparedState = sglmm.prepare(BlockParameter([latent, Vector(np.zeros(2))]))

    with pytest.raises(ValueError, match="Out-of-sample features required for prediction."):
        sglmm.create_predictor(preparedState, queryGrid)

def test_sglmm_predictor_shape():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.direct(grid, covFcn)
    
    X = np.random.randn(100, 2)
    sglmm = SGLMM(gp, grid, features=X)
    
    queryGrid = UniformGrid(0., 1., 20)
    X_pred_wrong = np.random.randn(10, 2)
    latent = gp.parameter.clone()
    latent.coordinate = np.zeros(gp.parameterDimension)
    preparedState = sglmm.prepare(BlockParameter([latent, Vector(np.zeros(2))]))

    with pytest.raises(ValueError, match="features must have shape"):
        sglmm.create_predictor(preparedState, queryGrid, features=X_pred_wrong)

def test_sglmm_predictor_missing_trend():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.direct(grid, covFcn)
    
    sglmm = SGLMM(gp, grid)
    queryGrid = UniformGrid(0., 1., 20)
    parameter = Vector(np.zeros(gp.parameterDimension))
    predictor = sglmm.create_predictor(sglmm.prepare(parameter), queryGrid)

    val = predictor.mean()
    assert val is not None

def test_sglmm_predictor_uses_explicit_fixed_effects():
    grid = UniformGrid(0., 1., 100)
    covFcn = MaternCovariance1D(1.0, 1.5, 1.5)
    gp = GaussianProcess.direct(grid, covFcn)
    
    X_obs = np.random.randn(100, 2)
    sglmm = SGLMM(gp, grid, features=X_obs)
    
    queryGrid = UniformGrid(0., 1., 20)
    X_pred = np.random.randn(20, 2)
    
    latent = Vector(np.random.randn(gp.parameterDimension))
    firstParameter = BlockParameter([latent, Vector(np.array([1.0, -0.5]))])
    secondParameter = BlockParameter([latent, Vector(np.array([2.0, 1.0]))])

    firstPredictor = sglmm.create_predictor(
        sglmm.prepare(firstParameter), queryGrid, features=X_pred)
    secondPredictor = sglmm.create_predictor(
        sglmm.prepare(secondParameter), queryGrid, features=X_pred)

    assert not np.allclose(firstPredictor.mean(), secondPredictor.mean())


def test_sglmm_predictor_snapshots_fixed_effects_and_features():
    grid = UniformGrid(0., 1., 20)
    gp = GaussianProcess.direct(
        grid, MaternCovariance1D(1.0, 1.5, 1.5))
    sglmm = SGLMM(gp, grid, features=np.zeros((20, 2)))
    queryGrid = UniformGrid(0., 1., 5)
    queryFeatures = np.arange(10, dtype=float).reshape(5, 2)
    latent = Vector(np.zeros(gp.parameterDimension))
    fixedEffect = Vector(np.array([1.0, -0.5]))
    parameter = BlockParameter([latent, fixedEffect])
    predictor = sglmm.create_predictor(
        sglmm.prepare(parameter), queryGrid, features=queryFeatures)
    expected = predictor.mean()

    latent.coordinate[:] = 1.0
    fixedEffect.coordinate[:] = 4.0
    queryFeatures[:] = -3.0
    returned = predictor.mean()
    returned[:] = 0.0

    np.testing.assert_array_equal(expected, predictor.mean())
