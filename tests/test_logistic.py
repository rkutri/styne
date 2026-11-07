import numpy as np
import pytest

from styne.parameter.vector import Vector
from styne.statistics.logistic import LogisticPosterior


def make_density():
    features = np.array([
        [1.0, -0.5],
        [0.2, 0.3],
        [-0.4, 0.7],
    ])
    return LogisticPosterior(features, np.array([1., 0., 1.]), 0.4)


def test_logistic_manual_gradient_matches_finite_difference():
    density = make_density()
    coordinate = np.array([0.1, -0.2])
    epsilon = 1e-6
    finiteDifference = np.array([
        (
            density.evaluate_log(Vector(
                coordinate + epsilon * np.eye(2)[index]
            ))
            - density.evaluate_log(Vector(
                coordinate - epsilon * np.eye(2)[index]
            ))
        ) / (2. * epsilon)
        for index in range(2)
    ])

    np.testing.assert_allclose(
        density.evaluate_log_gradient(Vector(coordinate)),
        finiteDifference,
        rtol=1e-6,
        atol=1e-6,
    )


def test_logistic_jax_autodiff_matches_manual_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    density = make_density()
    coordinate = jnp.array([0.1, -0.2])

    gradient = jax.jit(jax.grad(
        lambda value: density.evaluate_log(Vector(value))
    ))(coordinate)

    np.testing.assert_allclose(
        gradient,
        density.evaluate_log_gradient(Vector(coordinate)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_logistic_pytorch_autodiff_matches_manual_gradient():
    torch = pytest.importorskip("torch")
    density = make_density()
    coordinate = torch.tensor(
        [0.1, -0.2], dtype=torch.float64, requires_grad=True
    )

    value = density.evaluate_log(Vector(coordinate))
    gradient = torch.autograd.grad(value, coordinate)[0]

    torch.testing.assert_close(
        gradient,
        density.evaluate_log_gradient(Vector(coordinate)),
    )
