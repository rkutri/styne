import numpy as np
import pytest

from styne.model.sglmm import SGLMM
from styne.gp.gaussianprocess import GaussianProcess
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid, Grid
from styne.parameter.vector import Vector

def _numerical_jacobian(f, x, epsilon=1e-6):
    n = len(x)
    m = len(f(x))
    J = np.zeros((m, n))
    for i in range(n):
        xPlus = x.copy()
        xPlus[i] += epsilon
        xMinus = x.copy()
        xMinus[i] -= epsilon
        J[:, i] = (f(xPlus) - f(xMinus)) / (2 * epsilon)
    return J

def test_sglmm_predictor_identical_sites():
    """Test SGLMMPredictor when GP sites exactly match observation sites."""
    obsGrid = UniformGrid(0.0, 1.0, 10)
    cov = MaternCovariance1D(0.2, 1.5, 1.0)
    gp = GaussianProcess.dna(cov, q=10, d=1)
    predictor = SGLMM(gp, obsGrid)
    

    
    parameter = gp.parameter.clone()
    parameter.coordinate = np.random.randn(parameter.dimension)
    
    v = np.random.randn(parameter.dimension)
    w = np.random.randn(len(obsGrid))
    
    vParam = parameter.clone()
    vParam.coordinate = v
    deriv = predictor.directional_derivative(vParam)
    
    assert deriv.shape == (len(obsGrid),)
    
    adj = predictor.adjoint_directional_derivative(w)
    assert adj.shape == (parameter.dimension,)
    
    innerFwd = np.dot(deriv, w)
    innerBwd = np.dot(v, adj)
    np.testing.assert_allclose(innerFwd, innerBwd, rtol=1e-5)




