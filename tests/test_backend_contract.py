import numpy as np
import pytest

from styne.backend import BackendUnavailableError, get_backend, infer_backend
from styne.model.linear import LinearForwardMap
from styne.parameter.vector import Vector
from styne.statistics.covariance import (
    DiagonalCovarianceMatrix,
    IIDCovarianceMatrix,
)
from styne.statistics.data import Data
from styne.statistics.gaussian import Gaussian, GaussianDensity


@pytest.fixture(params=("numpy", "pytorch", "jax"))
def backend(request):
    try:
        return get_backend(request.param)
    except BackendUnavailableError:
        pytest.skip(f"{request.param} is not installed")


def test_vector_shared_backend_contract(backend):
    coordinate = backend.asarray(
        [[1.0, -2.0], [0.0, 2.0]], dtype="float32"
    )

    parameter = Vector(coordinate)
    replacement = parameter.with_coordinate(coordinate + 1.0)

    assert parameter.coordinate is coordinate
    assert infer_backend(parameter.coordinate) is backend
    assert parameter.backendMetadata == backend.metadata(coordinate)
    assert parameter.coordinate.shape == (2, 2)
    assert parameter.dimension == 2
    np.testing.assert_allclose(replacement.coordinate, coordinate + 1.0)

    with pytest.raises(ValueError, match="shape"):
        Vector(backend.zeros((1, 2, 3), dtype="float32"))


def test_data_shared_backend_contract(backend):
    design = backend.asarray([[0.0], [1.0], [2.0]], dtype="float32")
    measurement = backend.asarray([[2.0], [3.0], [5.0]], dtype="float32")
    data = Data(1, design)
    data.measurement = measurement

    assert data.design is design
    assert data.measurement is measurement
    assert infer_backend(data.design) is backend
    assert data.design.dtype == design.dtype
    assert data.measurement.shape == (3, 1)
    assert data.size == 3

    with pytest.raises(ValueError, match="incompatible"):
        data.measurement = backend.zeros((2, 2), dtype="float32")


def test_covariance_shared_backend_contract(backend):
    variances = backend.asarray([2.0, 0.5], dtype="float32")
    points = backend.asarray([[1.0, -2.0], [2.0, 3.0]], dtype="float32")
    covariance = DiagonalCovarianceMatrix(variances, scaling=1.5)

    applied = covariance.apply(points)
    inverse = covariance.apply_inverse(points)
    replacement = covariance.with_scaling(2.0)

    assert infer_backend(applied) is backend
    assert applied.dtype == points.dtype
    assert applied.shape == points.shape
    assert covariance.log_determinant().shape == ()
    assert covariance.quadratic_form(points).shape == (2,)
    assert covariance.dual_quadratic_form(points).shape == (2,)
    np.testing.assert_allclose(
        applied, [[3.0, -1.5], [6.0, 2.25]], rtol=1e-6
    )
    np.testing.assert_allclose(
        inverse, [[1 / 3, -8 / 3], [2 / 3, 4.0]], rtol=1e-6
    )
    np.testing.assert_allclose(covariance.scaling, 1.5, rtol=1e-6)
    np.testing.assert_allclose(replacement.scaling, 2.0, rtol=1e-6)

    with pytest.raises(ValueError, match="one-dimensional"):
        DiagonalCovarianceMatrix(backend.zeros((2, 2), dtype="float32"))


def test_linear_forward_map_shared_backend_contract(backend):
    features = backend.asarray(
        [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]], dtype="float32"
    )
    coordinate = backend.asarray(
        [[0.3, -0.4], [1.0, 2.0]], dtype="float32"
    )
    forwardMap = LinearForwardMap(features)

    evaluation = forwardMap(Vector(coordinate))

    assert infer_backend(evaluation) is backend
    assert evaluation.dtype == coordinate.dtype
    assert evaluation.shape == (2, 3)
    np.testing.assert_allclose(evaluation, coordinate @ features.T)

    with pytest.raises(ValueError, match="2D"):
        LinearForwardMap(backend.zeros(3, dtype="float32"))


def test_gaussian_density_shared_backend_contract(backend):
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


def test_gaussian_measure_shared_backend_contract(backend):
    mean = Vector(backend.zeros(2, dtype="float32"))
    measure = Gaussian(
        IIDCovarianceMatrix(2, backend.asarray(1.5, dtype="float32")), mean
    )
    randomState = backend.random_state(17)

    sample, nextState = measure.sample(randomState)

    assert infer_backend(sample.coordinate) is backend
    assert sample.coordinate.dtype == mean.coordinate.dtype
    assert sample.coordinate.shape == (2,)
    if backend.name == "jax":
        assert nextState is not randomState
    else:
        assert nextState is randomState


def test_gaussian_updates_are_functional(backend):
    mean = Vector(backend.zeros(2, dtype="float32"))
    covariance = IIDCovarianceMatrix(
        2, backend.asarray(1.5, dtype="float32")
    )
    measure = Gaussian(covariance, mean)
    replacementMean = Vector(backend.ones(2, dtype="float32"))
    replacementCovariance = IIDCovarianceMatrix(
        2, backend.asarray(0.5, dtype="float32")
    )

    withMean = measure.with_mean(replacementMean)
    withCovariance = measure.with_covariance(replacementCovariance)

    assert measure.mean is mean
    assert measure.covariance is covariance
    assert withMean.mean is replacementMean
    assert withCovariance.covariance is replacementCovariance
    with pytest.raises(AttributeError):
        measure.mean = replacementMean
    with pytest.raises(AttributeError):
        measure.covariance = replacementCovariance


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


def test_pytorch_gaussian_sample_retains_mean_gradient():
    torch = pytest.importorskip("torch")
    backend = get_backend("pytorch")
    meanCoordinate = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    measure = Gaussian(
        IIDCovarianceMatrix(2, torch.tensor(1.5, dtype=torch.float64)),
        Vector(meanCoordinate),
    )

    sample, _ = measure.sample(backend.random_state(9))
    gradient, = torch.autograd.grad(sample.coordinate.sum(), meanCoordinate)

    torch.testing.assert_close(gradient, torch.ones_like(meanCoordinate))


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
