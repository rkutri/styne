import numpy as np
import pytest

from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import GaussianDensity
from styne.statistics.mixture import GaussianMixtureDensity


def make_density(array):
    components = [
        GaussianDensity(
            IIDCovarianceMatrix(1, array(0.7)),
            Vector(array([-0.5])),
        ),
        GaussianDensity(
            IIDCovarianceMatrix(1, array(1.2)),
            Vector(array([1.0])),
        ),
    ]
    return GaussianMixtureDensity(components, weights=[0.3, 0.7])


def test_mixture_jax_autodiff_matches_manual_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    density = make_density(jnp.asarray)
    coordinate = jnp.array([0.2])

    gradient = jax.jit(jax.grad(
        lambda value: density.evaluate_log(Vector(value))
    ))(coordinate)

    np.testing.assert_allclose(
        gradient,
        density.evaluate_log_gradient(Vector(coordinate)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_mixture_pytorch_autodiff_matches_manual_gradient():
    torch = pytest.importorskip("torch")
    density = make_density(
        lambda value: torch.as_tensor(value, dtype=torch.float64)
    )
    coordinate = torch.tensor(
        [0.2], dtype=torch.float64, requires_grad=True
    )

    value = density.evaluate_log(Vector(coordinate))
    gradient = torch.autograd.grad(value, coordinate)[0]

    torch.testing.assert_close(
        gradient,
        density.evaluate_log_gradient(Vector(coordinate)),
    )
