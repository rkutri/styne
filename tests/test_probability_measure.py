import numpy as np
import pytest

from styne.backend import get_backend
from styne.parameter import Vector
from styne.statistics.dirac import DiracMeasure
from styne.statistics.gaussian import Gaussian
from styne.statistics.covariance import IIDCovarianceMatrix


def gaussian():
    return Gaussian(IIDCovarianceMatrix(2, 1.5), Vector(np.zeros(2)))


def test_gaussian_sample_propagates_numpy_random_state():
    measure = gaussian()
    firstState = get_backend("numpy").random_state(123)
    secondState = get_backend("numpy").random_state(123)

    first, nextFirstState = measure.sample(firstState)
    second, nextSecondState = measure.sample(secondState)

    np.testing.assert_allclose(first.coordinate, second.coordinate)
    assert nextFirstState is firstState
    assert nextSecondState is secondState


def test_generate_realisation_accepts_explicit_backend_state():
    measure = gaussian()
    state = get_backend("numpy").random_state(9)

    sample = measure.generate_realisation(randomState=state)

    assert isinstance(sample, Vector)
    assert sample.coordinate.shape == (2,)


def test_dirac_sample_does_not_consume_random_state():
    measure = DiracMeasure()
    measure.location = Vector(np.array([1.0, -2.0]))
    state = get_backend("numpy").random_state(7)

    sample, nextState = measure.sample(state)

    assert sample is measure.location
    assert nextState is state


def test_gaussian_sample_preserves_pytorch_state_and_arrays():
    torch = pytest.importorskip("torch", reason="PyTorch is optional")
    coordinate = torch.zeros(2, dtype=torch.float32)
    measure = Gaussian(
        IIDCovarianceMatrix(2, torch.tensor(1.5)), Vector(coordinate)
    )
    state = get_backend("pytorch").random_state(123)

    sample, nextState = measure.sample(state)

    assert isinstance(sample.coordinate, torch.Tensor)
    assert sample.coordinate.dtype == coordinate.dtype
    assert nextState is state


def test_gaussian_sample_propagates_jax_random_state():
    jnp = pytest.importorskip("jax.numpy", reason="JAX is optional")
    measure = Gaussian(
        IIDCovarianceMatrix(2, jnp.array(1.5)), Vector(jnp.zeros(2))
    )
    backend = get_backend("jax")
    state = backend.random_state(123)

    sample, nextState = measure.sample(state)

    assert type(sample.coordinate) is type(jnp.zeros(2))
    assert not np.array_equal(state, nextState)
