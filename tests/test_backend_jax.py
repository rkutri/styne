import numpy as np
import pytest


jax = pytest.importorskip("jax", reason="JAX is an optional backend")
jnp = pytest.importorskip("jax.numpy", reason="JAX is an optional backend")

from styne.backend import (  # noqa: E402
    BackendCapabilityError,
    get_backend,
    infer_backend,
)
from styne.backend.jax import JAXBackend, JAXNamespace  # noqa: E402
from styne.model.representation.expansion import Expansion  # noqa: E402
from styne.parameter import (  # noqa: E402
    BlockParameter,
    Function,
    Scalar,
    Vector,
)


class StaticExpansion(Expansion):

    @property
    def dimension(self):
        return 2

    def _bind(self, grid):
        raise NotImplementedError


def test_jax_backend_is_registered_for_arrays_and_tracers():
    backend = get_backend("jax")
    array = jnp.zeros(2, dtype=jnp.float32)

    assert isinstance(backend, JAXBackend)
    assert isinstance(backend.namespace, JAXNamespace)
    assert infer_backend(array) is backend
    assert backend.metadata(array).dtype == jnp.dtype("float32")
    assert backend.metadata(array).device == array.device

    inferredInsideJit = backend.compile(
        lambda value: 1 if infer_backend(value) is backend else 0
    )(array)
    assert inferredInsideJit == 1


def test_jax_construction_preserves_dtype_and_device():
    backend = get_backend("jax")
    device = jax.devices("cpu")[0]
    original = jnp.array([1.0, 2.0], dtype=jnp.float32)

    assert backend.asarray(original) is original
    assert backend.asarray(
        [1.0], dtype="float32"
    ).dtype == jnp.dtype("float32")
    assert backend.zeros(2, dtype="float32", device=device).device == device
    assert backend.ones(2, dtype="float32", device="cpu").device == device
    np.testing.assert_array_equal(backend.full(2, 3.0), [3.0, 3.0])
    np.testing.assert_array_equal(backend.eye(2), np.eye(2))
    np.testing.assert_array_equal(backend.arange(3), [0, 1, 2])
    np.testing.assert_array_equal(backend.arange(1, 4), [1, 2, 3])
    np.testing.assert_allclose(
        backend.linspace(0.0, 1.0, 3), [0.0, 0.5, 1.0]
    )


def test_jax_namespace_smoke_and_special_functions():
    namespace = get_backend("jax").namespace
    array = jnp.array([[-1.0, 2.0], [3.0, 4.0]])
    positive = jnp.array([1.0, 2.0, 3.0])

    np.testing.assert_array_equal(namespace.abs(array), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(
        namespace.all(array > 0, axis=1), [False, True]
    )
    np.testing.assert_array_equal(
        namespace.any(array < 0, axis=1), [True, False]
    )
    assert namespace.broadcast_to(positive, (2, 3)).shape == (2, 3)
    np.testing.assert_array_equal(
        namespace.clip(array, 0.0, 3.0), [[0, 2], [3, 3]]
    )
    assert namespace.concatenate((array, array)).shape == (4, 2)
    np.testing.assert_array_equal(namespace.diagonal(array), [-1, 4])
    assert namespace.expand_dims(array, axis=0).shape == (1, 2, 2)
    np.testing.assert_array_equal(
        namespace.isfinite(jnp.array([1.0, jnp.inf])), [True, False]
    )
    np.testing.assert_allclose(
        namespace.exp(namespace.log(positive)), positive
    )
    np.testing.assert_allclose(
        namespace.logaddexp(positive, 0.0),
        jnp.logaddexp(positive, 0.0),
    )
    np.testing.assert_array_equal(
        namespace.maximum(array, 0.0), [[0, 2], [3, 4]]
    )
    np.testing.assert_array_equal(
        namespace.minimum(array, 0.0), [[-1, 0], [0, 0]]
    )
    assert namespace.mean(positive).shape == ()
    np.testing.assert_allclose(
        namespace.norm(array, axis=1), jnp.linalg.norm(array, axis=1)
    )
    assert namespace.prod(positive).shape == ()
    np.testing.assert_allclose(namespace.sigmoid(jnp.array([0.0])), [0.5])
    np.testing.assert_allclose(namespace.sqrt(positive), jnp.sqrt(positive))
    np.testing.assert_array_equal(namespace.square(positive), [1, 4, 9])
    assert namespace.stack((array, array)).shape == (2, 2, 2)
    assert namespace.sum(positive).shape == ()
    np.testing.assert_array_equal(
        namespace.where(array > 0, array, 0.0), [[0, 2], [3, 4]]
    )
    np.testing.assert_allclose(
        namespace.gamma(positive), [1, 1, 2], rtol=1e-6
    )
    np.testing.assert_allclose(
        namespace.gammaln(positive),
        jnp.log(jnp.array([1, 1, 2])),
        atol=1e-6,
    )
    assert namespace.logsumexp(jnp.log(positive)).shape == ()

    with pytest.raises(BackendCapabilityError, match="arbitrary-order"):
        namespace.bessel_kv(0.5, positive)


def test_jax_random_state_is_explicit_and_transformable():
    backend = get_backend("jax")
    with pytest.raises(ValueError, match="explicit seed"):
        backend.random_state()

    firstState = backend.random_state(9182)
    secondState = backend.random_state(9182)
    firstNormal, firstState = backend.normal(
        firstState, (2, 3), dtype="float32"
    )
    secondNormal, secondState = backend.normal(
        secondState, (2, 3), dtype="float32"
    )
    np.testing.assert_array_equal(firstNormal, secondNormal)
    assert not np.array_equal(firstState, backend.random_state(9182))

    def draw_all(randomState):
        uniform, randomState = backend.uniform(randomState, 4, dtype="float32")
        poisson, randomState = backend.poisson(randomState, 2.0, shape=(4,))
        binomial, randomState = backend.binomial(
            randomState, 5, 0.4, shape=(4,)
        )
        return uniform, poisson, binomial, randomState

    first = backend.compile(draw_all)(firstState)
    second = backend.compile(draw_all)(secondState)
    for firstValue, secondValue in zip(first[:-1], second[:-1]):
        np.testing.assert_array_equal(firstValue, secondValue)
    np.testing.assert_array_equal(
        jax.random.key_data(first[-1]), jax.random.key_data(second[-1])
    )


def test_jax_automatic_differentiation_contract():
    backend = get_backend("jax")
    x = jnp.array([1.0, 2.0], dtype=jnp.float32)

    def function(value):
        return backend.namespace.sum(value ** 3)

    np.testing.assert_allclose(backend.grad(function)(x), [3.0, 12.0])
    value, gradient = backend.value_and_grad(function)(x)
    assert value == pytest.approx(9.0)
    np.testing.assert_allclose(gradient, [3.0, 12.0])
    np.testing.assert_allclose(
        backend.jacobian(lambda value: value ** 2)(x),
        [[2.0, 0.0], [0.0, 4.0]],
    )
    np.testing.assert_allclose(
        backend.hessian(function)(x), [[6.0, 0.0], [0.0, 12.0]]
    )
    primal, tangent = backend.jvp(
        lambda value: value ** 2, (x,), (jnp.ones_like(x),)
    )
    np.testing.assert_allclose(primal, [1.0, 4.0])
    np.testing.assert_allclose(tangent, [2.0, 4.0])
    primal, pullback = backend.vjp(lambda value: value ** 2, x)
    np.testing.assert_allclose(primal, [1.0, 4.0])
    np.testing.assert_allclose(pullback(jnp.ones_like(x))[0], [2.0, 4.0])


def test_jax_compilation_vectorisation_control_flow_and_scan():
    backend = get_backend("jax")
    compiled = backend.compile(lambda value: value * value + 1.0)
    vectorised = backend.vectorize(lambda value: value * value)

    np.testing.assert_allclose(compiled(jnp.array([2.0, 3.0])), [5.0, 10.0])
    np.testing.assert_allclose(
        vectorised(jnp.arange(4.0)), [0.0, 1.0, 4.0, 9.0]
    )
    assert backend.compile(
        lambda predicate: backend.cond(
            predicate, lambda x: x + 1, lambda x: x - 1, 2
        )
    )(jnp.array(True)) == 3

    def accumulate(carry, value):
        nextCarry = carry + value
        return nextCarry, nextCarry

    carry, values = backend.compile(
        lambda inputs: backend.scan(accumulate, 0.0, inputs)
    )(jnp.arange(4.0))
    assert carry == pytest.approx(6.0)
    np.testing.assert_allclose(values, [0.0, 1.0, 3.0, 6.0])


def test_jax_linear_algebra_and_spectral_transforms():
    backend = get_backend("jax")
    matrix = jnp.array([[2.0, 0.3], [0.3, 1.1]], dtype=jnp.float32)
    coordinate = jnp.array([0.4, -1.2], dtype=jnp.float32)
    chol = backend.cholesky(matrix)

    np.testing.assert_allclose(chol @ chol.T, matrix, atol=1e-6)
    np.testing.assert_allclose(
        backend.solve(matrix, coordinate),
        [0.3791469, -1.1943128],
        atol=1e-6,
    )
    lowerSolution = backend.solve_triangular(chol, coordinate, lower=True)
    np.testing.assert_allclose(chol @ lowerSolution, coordinate, atol=1e-6)

    array = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    np.testing.assert_allclose(
        backend.dct1(array), [8.0, -2.0, 0.0], atol=1e-5
    )
    np.testing.assert_allclose(
        backend.dst1(array),
        [4.0 + 4.0 * np.sqrt(2.0), -4.0, 4.0 * np.sqrt(2.0) - 4.0],
        atol=1e-5,
    )
    transformGradient = backend.grad(
        lambda value: backend.namespace.sum(backend.dct1(value))
    )(array)
    np.testing.assert_allclose(transformGradient, [3.0, 0.0, 1.0], atol=1e-5)


def test_jax_capabilities_are_explicit():
    capabilities = get_backend("jax").capabilities

    assert capabilities.automaticDifferentiation
    assert capabilities.compilation
    assert capabilities.vectorisation
    assert capabilities.controlFlow
    assert capabilities.spectralTransforms
    assert capabilities.transformedLoops


def test_jax_parameter_containers_transform_as_pytrees():
    expansion = StaticExpansion()
    parameter = BlockParameter(
        [
            Vector(jnp.array([
                [1.0, 2.0],
                [3.0, 4.0],
            ])),
            Scalar(jnp.array([[5.0], [6.0]])),
            Function(jnp.array([
                [7.0, 8.0],
                [9.0, 10.0],
            ]), expansion),
        ],
        {"vector": 0, "scalar": 1, "function": 2},
    )

    leaves, structure = jax.tree_util.tree_flatten(parameter)
    reconstructed = jax.tree_util.tree_unflatten(structure, leaves)

    assert len(leaves) == 3
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert reconstructed.names == parameter.names
    assert reconstructed["function"].expansion is expansion

    compiled = jax.jit(
        lambda state: state.with_coordinate(state.coordinate * 2.0)
    )(parameter)
    vectorised = jax.vmap(
        lambda state: state.with_coordinate(state.coordinate + 1.0)
    )(parameter)
    gradient = jax.grad(
        lambda state: jnp.sum(state.coordinate ** 2)
    )(parameter)

    np.testing.assert_allclose(compiled.coordinate, 2.0 * parameter.coordinate)
    np.testing.assert_allclose(
        vectorised.coordinate, parameter.coordinate + 1.0
    )
    np.testing.assert_allclose(
        gradient.coordinate, 2.0 * parameter.coordinate
    )
    assert compiled["function"].expansion is expansion
    assert vectorised.names == parameter.names


def test_jax_parameter_pytrees_accept_structural_placeholders():
    expansion = StaticExpansion()
    parameter = BlockParameter(
        [
            Vector(jnp.array([1.0, 2.0])),
            Scalar(jnp.array([3.0])),
            Function(jnp.array([4.0, 5.0]), expansion),
        ],
        {"vector": 0, "scalar": 1, "function": 2},
    )
    leaves, structure = jax.tree_util.tree_flatten(parameter)
    placeholders = [object() for leaf in leaves]

    reconstructed = jax.tree_util.tree_unflatten(structure, placeholders)

    assert reconstructed.names == parameter.names
    assert reconstructed.dimension == parameter.dimension
    assert reconstructed["function"].expansion is expansion
    assert all(
        reconstructed.block(index).coordinate is placeholder
        for index, placeholder in enumerate(placeholders)
    )
