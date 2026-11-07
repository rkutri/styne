import importlib

import numpy as np
import pytest

import styne.utility as utilityModule
from styne.model.forwardmap import ForwardMap
from styne.parameter.vector import Vector
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.response import GaussianResponse


class DifferentiableMock(ForwardMap):
    """Identity model that satisfies DifferentiableForwardMap."""

    def __init__(self, dim=2):
        super().__init__()
        self._dim = dim
        self.evaluations = 0

    @property
    def pType(self):
        return Vector

    @property
    def pDim(self):
        return self._dim

    def _prepare(self, parameter):
        return parameter.coordinate

    def _evaluate(self, preparedState):
        self.evaluations += 1
        return preparedState

    def directional_derivative(self, parameter, direction):
        return direction

    def adjoint_derivative(self, parameter, cotangent):
        return np.asarray(cotangent)


class NonDifferentiableMock(ForwardMap):

    @property
    def pType(self):
        return Vector

    @property
    def pDim(self):
        return 2

    def _prepare(self, parameter):
        return parameter.coordinate

    def _evaluate(self, preparedState):
        return np.zeros(2)


def test_initialisation(mock_likelihood, mock_data, mock_forward_model):
    assert mock_likelihood.data is mock_data
    assert mock_likelihood.model is mock_forward_model
    assert isinstance(mock_likelihood.response, GaussianResponse)
    assert mock_likelihood.domainType is Vector
    assert mock_likelihood.domainDimension == 2


def test_generic_evaluation_cache_is_removed():
    assert "EvaluationCache" not in utilityModule.__all__
    assert not hasattr(utilityModule, "EvaluationCache")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("styne.utility.memoisation")


def test_log_likelihood_does_not_cache_parameter_evaluations(
        mock_data, mock_noise):
    model = DifferentiableMock(dim=2)
    likelihood = RegressionLikelihood(
        mock_data,
        model,
        GaussianResponse(mock_noise.density.covariance),
    )
    parameter = Vector(np.array([0.5, 0.5]))
    first = likelihood.evaluate_log(parameter)
    second = likelihood.evaluate_log(parameter)

    assert first == second
    assert model.evaluations == 2
    assert not hasattr(likelihood, "_logLikelihoodCache")


def test_log_gradient_does_not_cache_parameter_evaluations(
        mock_data, mock_noise):
    model = DifferentiableMock(dim=2)
    likelihood = RegressionLikelihood(
        mock_data,
        model,
        GaussianResponse(mock_noise.density.covariance),
    )
    parameter = Vector(np.array([0.5, 0.5]))
    first = likelihood.evaluate_log_gradient(parameter)
    second = likelihood.evaluate_log_gradient(parameter)

    np.testing.assert_array_equal(first, second)
    assert model.evaluations == 2
    assert not hasattr(likelihood, "_gradientCache")


def test_non_differentiable_model_exception(mock_data, mock_noise):
    likelihood = RegressionLikelihood(
        mock_data,
        NonDifferentiableMock(),
        GaussianResponse(mock_noise.density.covariance),
    )
    parameter = Vector(np.array([0.5, 0.5]))
    with pytest.raises(RuntimeError):
        likelihood.evaluate_log_gradient(parameter)
