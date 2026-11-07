import numpy as np
import pytest

from styne.backend import (
    BackendCapabilityError,
    get_backend,
    infer_backend,
)
from styne.backend.numpy import NumPyBackend, NumPyNamespace


def test_numpy_backend_is_registered_for_arrays_and_scalars():
    backend = get_backend("numpy")

    assert isinstance(backend, NumPyBackend)
    assert infer_backend(np.zeros(2)) is backend
    assert infer_backend(np.float64(1.0)) is backend
    assert isinstance(backend.namespace, NumPyNamespace)


def test_numpy_metadata_and_construction_preserve_dtype():
    backend = get_backend("numpy")
    original = np.array([1.0, 2.0], dtype=np.float32)

    assert backend.asarray(original) is original
    assert backend.metadata(original).dtype == np.dtype("float32")
    assert backend.metadata(original).device == "cpu"
    assert backend.zeros(2, dtype="float32").dtype == np.dtype("float32")
    assert backend.ones(2, dtype="float64").dtype == np.dtype("float64")
    np.testing.assert_array_equal(backend.full(2, 3.0), [3.0, 3.0])
    np.testing.assert_array_equal(backend.eye(2), np.eye(2))
    np.testing.assert_array_equal(backend.arange(3), [0, 1, 2])
    np.testing.assert_array_equal(backend.arange(1, 4), [1, 2, 3])
    np.testing.assert_array_equal(
        backend.linspace(0.0, 1.0, 3), [0.0, 0.5, 1.0]
    )


def test_numpy_backend_rejects_non_cpu_devices_and_non_arrays():
    backend = get_backend("numpy")

    with pytest.raises(ValueError, match="only the CPU device"):
        backend.zeros(2, device="cuda")
    with pytest.raises(TypeError, match="metadata requires"):
        backend.metadata([1.0, 2.0])


def test_numpy_namespace_array_operations():
    namespace = get_backend("numpy").namespace
    array = np.array([[-1.0, 2.0], [3.0, 4.0]])

    np.testing.assert_array_equal(
        namespace.abs(array), [[1.0, 2.0], [3.0, 4.0]]
    )
    np.testing.assert_array_equal(
        namespace.all(array > 0, axis=1), [False, True]
    )
    np.testing.assert_array_equal(
        namespace.any(array < 0, axis=1), [True, False]
    )
    np.testing.assert_array_equal(
        namespace.broadcast_to(np.array([1.0, 2.0]), (2, 2)),
        [[1.0, 2.0], [1.0, 2.0]],
    )
    np.testing.assert_array_equal(
        namespace.clip(array, 0.0, 3.0), [[0, 2], [3, 3]]
    )
    np.testing.assert_array_equal(
        namespace.concatenate((array, array), axis=0),
        np.concatenate((array, array), axis=0),
    )
    np.testing.assert_array_equal(
        namespace.diagonal(array), [-1.0, 4.0]
    )
    assert namespace.expand_dims(array, axis=0).shape == (1, 2, 2)
    np.testing.assert_array_equal(
        namespace.isfinite([1.0, np.inf]), [True, False]
    )
    np.testing.assert_array_equal(
        namespace.maximum(array, 0.0), [[0, 2], [3, 4]]
    )
    np.testing.assert_array_equal(
        namespace.minimum(array, 0.0), [[-1, 0], [0, 0]]
    )
    np.testing.assert_allclose(
        namespace.norm(array, axis=1), np.linalg.norm(array, axis=1)
    )
    np.testing.assert_array_equal(
        namespace.prod(array, axis=1), [-2.0, 12.0]
    )
    np.testing.assert_array_equal(
        namespace.square(array), [[1, 4], [9, 16]]
    )
    assert namespace.stack((array, array)).shape == (2, 2, 2)
    np.testing.assert_array_equal(
        namespace.where(array > 0, array, 0.0), [[0, 2], [3, 4]]
    )


def test_numpy_namespace_scalar_and_special_operations():
    namespace = get_backend("numpy").namespace
    positive = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(
        namespace.exp(namespace.log(positive)), positive
    )
    np.testing.assert_allclose(
        namespace.logaddexp(positive, 0.0),
        np.logaddexp(positive, 0.0),
    )
    assert namespace.mean(positive) == np.float64(2.0)
    assert namespace.sum(positive) == np.float64(6.0)
    np.testing.assert_allclose(namespace.sqrt(positive), np.sqrt(positive))
    np.testing.assert_allclose(namespace.sigmoid([0.0]), [0.5])
    np.testing.assert_allclose(namespace.gamma(positive), [1.0, 1.0, 2.0])
    np.testing.assert_allclose(
        namespace.gammaln(positive), np.log([1.0, 1.0, 2.0])
    )
    np.testing.assert_allclose(
        namespace.bessel_kv(0.5, positive),
        np.sqrt(np.pi / (2.0 * positive)) * np.exp(-positive),
    )
    assert namespace.logsumexp(
        np.log([1.0, 2.0, 3.0])
    ) == pytest.approx(np.log(6.0))


def test_numpy_random_operations_are_seeded_and_propagate_state():
    backend = get_backend("numpy")
    firstState = backend.random_state(9182)
    secondState = backend.random_state(9182)

    firstNormal, returnedState = backend.normal(
        firstState, (2, 3), dtype="float32"
    )
    secondNormal, secondState = backend.normal(
        secondState, (2, 3), dtype="float32"
    )
    assert returnedState is firstState
    np.testing.assert_array_equal(firstNormal, secondNormal)
    assert firstNormal.dtype == np.dtype("float32")

    firstUniform, firstState = backend.uniform(firstState, 4)
    secondUniform, secondState = backend.uniform(secondState, 4)
    np.testing.assert_array_equal(firstUniform, secondUniform)

    firstPoisson, firstState = backend.poisson(firstState, 2.0, shape=4)
    secondPoisson, secondState = backend.poisson(secondState, 2.0, shape=4)
    np.testing.assert_array_equal(firstPoisson, secondPoisson)

    firstBinomial, _ = backend.binomial(firstState, 5, 0.4, shape=4)
    secondBinomial, _ = backend.binomial(secondState, 5, 0.4, shape=4)
    np.testing.assert_array_equal(firstBinomial, secondBinomial)


def test_numpy_linear_algebra_matches_frozen_covariance_reference():
    backend = get_backend("numpy")
    matrix = backend.asarray([[2.0, 0.3], [0.3, 1.1]])
    coordinate = backend.asarray([0.4, -1.2])
    chol = backend.cholesky(matrix)

    np.testing.assert_allclose(chol @ chol.T, matrix, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        backend.solve(matrix, coordinate),
        [0.37914691943127965, -1.1943127962085307],
        rtol=0.0,
        atol=1e-12,
    )
    lowerSolution = backend.solve_triangular(chol, coordinate, lower=True)
    np.testing.assert_allclose(
        chol @ lowerSolution, coordinate, rtol=0.0, atol=1e-12
    )


def test_numpy_spectral_transforms_match_type_one_reference_values():
    backend = get_backend("numpy")
    array = backend.asarray([1.0, 2.0, 3.0])

    np.testing.assert_allclose(
        backend.dct1(array), [8.0, -2.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        backend.dst1(array),
        [4.0 + 4.0 * np.sqrt(2.0), -4.0, 4.0 * np.sqrt(2.0) - 4.0],
        atol=1e-12,
    )


def test_numpy_control_flow_and_capabilities_are_explicit():
    backend = get_backend("numpy")

    assert backend.cond(
        np.array(True), lambda x: x + 1, lambda x: x - 1, 2
    ) == 3
    assert backend.capabilities.controlFlow
    assert backend.capabilities.spectralTransforms
    assert not backend.capabilities.automaticDifferentiation
    assert not backend.capabilities.compilation
    assert not backend.capabilities.vectorisation
    assert not backend.capabilities.transformedLoops

    with pytest.raises(
            BackendCapabilityError, match="automatic differentiation"):
        backend.grad(lambda value: value)
    with pytest.raises(BackendCapabilityError, match="compilation"):
        backend.compile(lambda value: value)
    with pytest.raises(BackendCapabilityError, match="vectorisation"):
        backend.vectorize(lambda value: value)
    with pytest.raises(BackendCapabilityError, match="transformed loops"):
        backend.scan(
            lambda state, value: (state + value, state), 0, np.arange(3)
        )
