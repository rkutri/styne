import numpy as np
import pytest

from styne.backend import BackendUnavailableError, get_backend, infer_backend
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import GaussianDensity


@pytest.mark.parametrize("backendName", ("numpy", "pytorch", "jax"))
def test_gaussian_density_shared_backend_contract(backendName):
    try:
        backend = get_backend(backendName)
    except BackendUnavailableError:
        pytest.skip(f"{backendName} is not installed")

    coordinate = backend.asarray(
        [[1.0, -2.0], [0.0, 2.0]], dtype="float32"
    )
    mean = Vector(backend.zeros(2, dtype="float32"))
    covariance = IIDCovarianceMatrix(
        2, backend.asarray(2.0, dtype="float32")
    )
    density = GaussianDensity(covariance, mean)

    value = density.evaluate_log(Vector(coordinate))

    assert infer_backend(value) is backend
    assert value.dtype == coordinate.dtype
    assert value.shape == (2,)
    np.testing.assert_allclose(value, [-1.25, -1.0], rtol=1e-6)


def test_pytorch_gaussian_density_has_first_and_second_derivatives():
    torch = pytest.importorskip("torch")
    backend = get_backend("pytorch")
    variance = torch.tensor(1.7, dtype=torch.float64)
    mean = Vector(torch.zeros(2, dtype=torch.float64))
    density = GaussianDensity(IIDCovarianceMatrix(2, variance), mean)

    def objective(coordinate):
        return density.evaluate_log(Vector(coordinate))

    coordinate = torch.tensor(
        [0.4, -0.7], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(objective, (coordinate,))
    assert torch.autograd.gradgradcheck(objective, (coordinate,))

    compiled = backend.compile(objective, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(coordinate), objective(coordinate))


def test_jax_gaussian_density_transforms_through_public_interface():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    variance = jnp.array(1.7, dtype=jnp.float32)
    mean = Vector(jnp.zeros(2, dtype=jnp.float32))
    density = GaussianDensity(IIDCovarianceMatrix(2, variance), mean)

    def objective(coordinate):
        return density.evaluate_log(Vector(coordinate))

    coordinate = jnp.array([0.4, -0.7], dtype=jnp.float32)
    gradient = jax.grad(objective)(coordinate)
    hessian = jax.hessian(objective)(coordinate)

    np.testing.assert_allclose(gradient, -coordinate / variance, rtol=1e-5)
    np.testing.assert_allclose(
        hessian, -jnp.eye(2, dtype=jnp.float32) / variance, rtol=1e-5
    )
    np.testing.assert_allclose(
        jax.jit(objective)(coordinate), objective(coordinate)
    )
    np.testing.assert_allclose(
        jax.vmap(objective)(jnp.stack((coordinate, -coordinate))),
        [objective(coordinate), objective(-coordinate)],
    )
