import numpy as np
import pytest

from styne.gp.gaussianprocess import GaussianProcess
from styne.utility.grid import UniformGrid
from styne.statistics.stationary import matern_fourier


class StubFourierCov:
    """
    A stub Fourier covariance class for testing DNA Gaussian Processes.
    """

    def __init__(self, lengthScale: float):
        self.lengthScale = lengthScale

    def evaluate_fourier(self, freqs: np.ndarray) -> np.ndarray:
        d = freqs.shape[1] if freqs.ndim > 1 else 1
        return matern_fourier(freqs, self.lengthScale, 1.5, 1.0, d=d)


@pytest.mark.parametrize("q, alpha", [
    ((20, 12), (1.0, 1.0)),
    ((16, 16), (1.0, 2.0)),
    ((24, 10), (1.0, 1.7)),
])
def test_jacobian_adjoint_rectangular(q, alpha):
    d = 2
    gp = GaussianProcess.dna(StubFourierCov(0.2), q, d, alpha)

    sites = UniformGrid((0.13, 0.87, 5), (0.11, 0.83, 7))
    evaluation = gp.bind(sites)

    n_spec = gp.parameterDimension
    n_obs = sites.to_array().shape[0]

    rng = np.random.default_rng(0)
    v = rng.standard_normal(n_spec)
    w = rng.standard_normal(n_obs)

    coefficient = np.zeros(n_spec)
    Jv = evaluation.directional_derivative(coefficient, v)
    JTw = evaluation.adjoint_derivative(coefficient, w)

    lhs = float(np.dot(np.ravel(Jv), w))
    rhs = float(np.dot(v, np.ravel(JTw)))
    assert abs(lhs - rhs) <= 1e-9 * (1.0 + abs(lhs)), (lhs, rhs)
