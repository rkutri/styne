import numpy as np
import pytest

from styne.gp.gaussianprocess import GaussianProcess
from styne.model.sglmm import SGLMM
from styne.parameter.block import BlockParameter
from styne.parameter.vector import Vector
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid


def test_dna_out_of_sample_evaluation_batches_coordinates():
    gp = GaussianProcess.dna(
        MaternCovariance1D(1.0, 1.5, 1.5), q=10, d=1
    )
    queryGrid = UniformGrid(0.0, 1.0, 20)
    coefficient = np.random.default_rng(1).standard_normal(
        (5, gp.parameterDimension)
    )

    prediction = gp.evaluate(coefficient, queryGrid)

    assert prediction.shape == (5, 20)


def sglmm_with_features():
    grid = UniformGrid(0.0, 1.0, 30)
    gp = GaussianProcess.direct(
        grid, MaternCovariance1D(1.0, 1.5, 1.5)
    )
    return gp, SGLMM(gp, grid, features=np.zeros((len(grid), 2)))


def test_sglmm_prediction_requires_out_of_sample_features():
    gp, model = sglmm_with_features()
    state = model.prepare(BlockParameter([
        Vector(np.zeros(gp.parameterDimension)),
        Vector(np.zeros(2)),
    ]))

    with pytest.raises(
            ValueError, match="Out-of-sample features required"):
        model.predict(state, UniformGrid(0.0, 1.0, 8))


def test_sglmm_prediction_validates_out_of_sample_feature_shape():
    gp, model = sglmm_with_features()
    state = model.prepare(BlockParameter([
        Vector(np.zeros(gp.parameterDimension)),
        Vector(np.zeros(2)),
    ]))

    with pytest.raises(ValueError, match="features must have shape"):
        model.predict(
            state,
            UniformGrid(0.0, 1.0, 8),
            features=np.zeros((7, 2)),
        )


def test_sglmm_prediction_composes_latent_and_fixed_effects():
    gp, model = sglmm_with_features()
    queryGrid = UniformGrid(0.0, 1.0, 8)
    queryFeatures = np.arange(16, dtype=float).reshape(8, 2)
    coefficient = np.linspace(-0.5, 0.8, gp.parameterDimension)
    fixedEffect = np.array([1.0, -0.5])
    state = model.prepare(BlockParameter([
        Vector(coefficient), Vector(fixedEffect)
    ]))

    prediction = model.predict(
        state, queryGrid, features=queryFeatures
    )

    np.testing.assert_allclose(
        prediction,
        gp.evaluate(coefficient, queryGrid)
        + queryFeatures @ fixedEffect,
    )
