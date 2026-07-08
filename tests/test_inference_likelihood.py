import numpy as np
import pytest

from styne.model.model import Model
from styne.parameter.vector import Vector
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.response import GaussianResponse


class _DifferentiableMock(Model):
    """Identity model that satisfies DifferentiableModel."""

    def __init__(self, dim=2):
        super().__init__()
        self._dim = dim

    @property
    def pType(self):
        return Vector

    @property
    def pDim(self):
        return self._dim

    def _interpolate(self, parameter):
        self._p = parameter

    def _evaluate(self):
        self._evaluation = self._p.coordinate

    def directional_derivative(self, parameter):
        return parameter.clone()

    def adjoint_directional_derivative(self, w):
        return np.asarray(w)


class _NonDifferentiableMock(Model):

    @property
    def pType(self):
        return Vector

    @property
    def pDim(self):
        return 2

    def _interpolate(self, parameter):
        pass

    def _evaluate(self):
        self._evaluation = np.zeros(2)


def test_initialisation(mock_likelihood, mock_data, mock_forward_model):
    assert mock_likelihood.data is mock_data
    assert mock_likelihood.model is mock_forward_model
    assert isinstance(mock_likelihood.response, GaussianResponse)
    assert mock_likelihood.domainType is Vector
    assert mock_likelihood.domainDimension == 2


def test_memoisation(mock_likelihood):
    parameter = Vector(np.array([0.5, 0.5]))
    logLFirst = mock_likelihood.evaluate_log(parameter)
    logLCached = mock_likelihood.evaluate_log(parameter)

    assert logLFirst == logLCached
    assert mock_likelihood._logLikelihoodCache.contains(parameter)
    assert mock_likelihood._logLikelihoodCache.retrieve(parameter) == logLFirst


def test_gradient_memoisation(mock_data, mock_noise):
    likelihood = RegressionLikelihood(
        mock_data,
        _DifferentiableMock(dim=2),
        GaussianResponse(mock_noise.density.covariance),
    )
    parameter = Vector(np.array([0.5, 0.5]))
    gradientFirst = likelihood.evaluate_log_gradient(parameter)
    gradientCached = likelihood.evaluate_log_gradient(parameter)

    np.testing.assert_array_equal(gradientFirst, gradientCached)
    assert likelihood._gradientCache.contains(parameter)


def test_non_differentiable_model_exception(mock_data, mock_noise):
    likelihood = RegressionLikelihood(
        mock_data,
        _NonDifferentiableMock(),
        GaussianResponse(mock_noise.density.covariance),
    )
    parameter = Vector(np.array([0.5, 0.5]))
    with pytest.raises(RuntimeError):
        likelihood.evaluate_log_gradient(parameter)


def test_condition_on_clears_caches(mock_data, mock_noise):
    model = _DifferentiableMock(dim=2)
    likelihood = RegressionLikelihood(
        mock_data, model, GaussianResponse(mock_noise.density.covariance)
    )
    parameter = Vector(np.array([0.5, 0.5]))
    likelihood.evaluate_log(parameter)
    likelihood.evaluate_log_gradient(parameter)

    assert likelihood._logLikelihoodCache.contains(parameter)
    assert likelihood._gradientCache.contains(parameter)
    assert likelihood.model.evaluation is not None

    likelihood.condition_on(parameter)

    assert not likelihood._logLikelihoodCache.contains(parameter)
    assert not likelihood._gradientCache.contains(parameter)
    assert likelihood.model.evaluation is None

