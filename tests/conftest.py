import pytest
import numpy as np
import random

@pytest.fixture(autouse=True)
def global_random_seed():
    np.random.seed(42)
    random.seed(42)

from styne.model.model import Model

class MockIdentityModel(Model):
    def __init__(self, dim=2):
        super().__init__()
        self._dim = dim

    @property
    def pType(self):
        from styne.parameter.vector import Vector
        return Vector

    @property
    def pDim(self):
        return self._dim

    def _interpolate(self, parameter):
        self._p = parameter

    def _evaluate(self):
        if hasattr(self, '_p'):
            self._evaluation = self._p.coordinate
        else:
            self._evaluation = np.zeros(self._dim)


@pytest.fixture
def mock_noise():
    from styne.statistics.gaussian import Gaussian
    from styne.statistics.covariance import IIDCovarianceMatrix
    return Gaussian(IIDCovarianceMatrix(2, 1.0))


@pytest.fixture
def mock_data():
    from styne.statistics.data import Data
    data = Data(2, np.zeros((1, 2)))
    data.measurement = np.zeros((1, 2))
    return data


@pytest.fixture
def mock_forward_model():
    return MockIdentityModel(dim=2)


@pytest.fixture
def mock_likelihood(mock_data, mock_forward_model, mock_noise):
    from styne.statistics.likelihood import RegressionLikelihood
    from styne.statistics.response import GaussianResponse
    return RegressionLikelihood(
        mock_data, mock_forward_model, GaussianResponse(mock_noise.density.covariance)
    )
