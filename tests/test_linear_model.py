import numpy as np
import pytest

from styne.model import DifferentiableForwardMap
from styne.model.linear import LinearForwardMap
from styne.parameter.vector import Vector
from styne.statistics.data import Data
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.response import PoissonResponse


def test_linear_forward_map_prepare_evaluate():
    features = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    model = LinearForwardMap(features)
    parameter = Vector(np.array([0.3, -0.4]))

    assert model.pDim == 2
    assert isinstance(parameter, model.pType)

    np.testing.assert_allclose(
        model(parameter), features @ parameter.coordinate,
        rtol=0.0, atol=1e-12,
    )

    assert isinstance(model, DifferentiableForwardMap)
    cotangent = np.array([0.5, -1.0, 2.0])
    np.testing.assert_allclose(
        model.adjoint_derivative(parameter, cotangent),
        cotangent @ features,
    )


def test_linear_forward_map_rejects_non_2d_features():
    with pytest.raises(ValueError):
        LinearForwardMap(np.array([1.0, 2.0, 3.0]))


def test_jax_linear_likelihood_gradient_matches_matrix_formula():
    jax = pytest.importorskip("jax", reason="JAX is optional")
    jnp = pytest.importorskip("jax.numpy", reason="JAX is optional")

    features = jnp.array([[1.0, -0.5], [0.2, 0.3], [-0.4, 0.7]])
    measurement = jnp.array([[2.0], [1.0], [3.0]])
    coordinate = jnp.array([0.1, -0.2])
    data = Data(1, features)
    data.measurement = measurement
    likelihood = RegressionLikelihood(
        data, LinearForwardMap(features), PoissonResponse()
    )

    gradient = jax.jit(jax.grad(
        lambda beta: likelihood.evaluate_log(Vector(beta))
    ))(coordinate)
    expected = features.T @ (
        measurement.reshape((-1,)) - jnp.exp(features @ coordinate)
    )

    np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-6)


def test_pytorch_linear_likelihood_gradient_matches_matrix_formula():
    torch = pytest.importorskip("torch", reason="PyTorch is optional")

    features = torch.tensor(
        [[1.0, -0.5], [0.2, 0.3], [-0.4, 0.7]], dtype=torch.float64
    )
    measurement = torch.tensor([[2.0], [1.0], [3.0]], dtype=torch.float64)
    coordinate = torch.tensor(
        [0.1, -0.2], dtype=torch.float64, requires_grad=True
    )
    data = Data(1, features)
    data.measurement = measurement
    likelihood = RegressionLikelihood(
        data, LinearForwardMap(features), PoissonResponse()
    )

    gradient = torch.autograd.grad(
        likelihood.evaluate_log(Vector(coordinate)), coordinate
    )[0]
    expected = features.T @ (
        measurement.reshape((-1,)) - torch.exp(features @ coordinate)
    )

    torch.testing.assert_close(gradient, expected)
