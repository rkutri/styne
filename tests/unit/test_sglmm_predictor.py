import numpy as np

from styne.gp.gaussianprocess import GaussianProcess
from styne.model.sglmm import SGLMM
from styne.parameter.vector import Vector
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid


def test_sglmm_evaluates_explicit_latent_coordinates():
    obsGrid = UniformGrid(0.0, 1.0, 10)
    gp = GaussianProcess.dna(
        MaternCovariance1D(0.2, 1.5, 1.0), q=10, d=1
    )
    model = SGLMM(gp, obsGrid)
    coordinate = np.linspace(-0.4, 0.3, gp.parameterDimension)

    evaluation = model(Vector(coordinate))

    np.testing.assert_allclose(evaluation, gp.evaluate(coordinate, obsGrid))
