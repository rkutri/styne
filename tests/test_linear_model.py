import numpy as np
import pytest

from styne.model.linear import LinearForwardMap
from styne.parameter.vector import Vector


def test_linear_forward_map_prepare_evaluate_and_jacobian():
    features = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    model = LinearForwardMap(features)
    parameter = Vector(np.array([0.3, -0.4]))

    assert model.pDim == 2
    assert isinstance(parameter, model.pType)

    np.testing.assert_allclose(
        model(parameter), features @ parameter.coordinate,
        rtol=0.0, atol=1e-12,
    )

    direction = Vector(np.array([1.0, 2.0]))
    np.testing.assert_allclose(
        model.directional_derivative(parameter, direction).coordinate,
        features @ direction.coordinate, rtol=0.0, atol=1e-12,
    )

    residual = np.array([0.2, -0.1, 0.7])
    np.testing.assert_allclose(
        model.adjoint_derivative(parameter, residual),
        features.T @ residual, rtol=0.0, atol=1e-12,
    )


def test_linear_forward_map_rejects_non_2d_features():
    with pytest.raises(ValueError):
        LinearForwardMap(np.array([1.0, 2.0, 3.0]))
