import numpy as np
import pytest

from styne.backend import MixedBackendError
from styne.parameter import BlockParameter, Scalar, Vector


def test_block_parameter_is_immutable_and_preserves_structure():
    first = Vector(np.array([1.0, 2.0], dtype=np.float32))
    second = Scalar(np.array([3.0], dtype=np.float32))
    names = {"field": 0, "scale": 1}
    parameter = BlockParameter([first, second], names)
    names["field"] = 1

    replacementCoordinate = np.array(
        [4.0, 5.0, 6.0], dtype=np.float32
    )
    replacement = parameter.with_coordinate(replacementCoordinate)

    assert BlockParameter.coordinate.fset is None
    assert not hasattr(parameter, "clone")
    assert parameter.block(0) is first
    assert parameter.block(1) is second
    assert parameter["field"] is first
    assert parameter.names == {"field": 0, "scale": 1}
    np.testing.assert_array_equal(parameter.coordinate, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        replacement.coordinate, replacementCoordinate
    )
    np.testing.assert_array_equal(replacement.block(0).coordinate, [4.0, 5.0])
    np.testing.assert_array_equal(replacement.block(1).coordinate, [6.0])


def test_block_parameter_broadcasts_and_reconstructs_batches():
    first = Vector(np.array([1.0, 2.0], dtype=np.float64))
    second = Scalar(np.array([[3.0], [4.0]], dtype=np.float64))
    parameter = BlockParameter([first, second])

    expected = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 4.0],
    ])
    np.testing.assert_array_equal(parameter.coordinate, expected)

    replacement = parameter.with_coordinate(expected + 1.0)

    assert replacement.block(0).coordinate.shape == (2, 2)
    assert replacement.block(1).coordinate.shape == (2, 1)
    np.testing.assert_array_equal(replacement.coordinate, expected + 1.0)
    np.testing.assert_array_equal(parameter.coordinate, expected)


def test_block_parameter_rejects_incompatible_batches():
    first = Vector(np.zeros((2, 2)))
    second = Vector(np.zeros((3, 1)))
    parameter = BlockParameter([first, second])

    with pytest.raises(ValueError, match="not broadcastable"):
        parameter.coordinate


def test_block_parameter_rejects_mixed_dtypes():
    with pytest.raises(ValueError, match="dtype and device"):
        BlockParameter([
            Vector(np.ones(2, dtype=np.float32)),
            Vector(np.ones(1, dtype=np.float64)),
        ])


def test_block_parameter_can_move_to_a_new_backend_with_coordinate():
    jnp = pytest.importorskip("jax.numpy", reason="JAX is optional")
    parameter = BlockParameter([
        Vector(np.array([1.0, 2.0], dtype=np.float32)),
        Scalar(np.array([3.0], dtype=np.float32)),
    ])

    replacement = parameter.with_coordinate(
        jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
    )

    assert replacement.backend.name == "jax"
    assert replacement.block(0).backend.name == "jax"
    assert replacement.block(1).backend.name == "jax"


def test_block_parameter_rejects_mixed_backends():
    jnp = pytest.importorskip("jax.numpy", reason="JAX is optional")

    with pytest.raises(MixedBackendError):
        BlockParameter([
            Vector(np.ones(2, dtype=np.float32)),
            Vector(jnp.ones(1, dtype=jnp.float32)),
        ])


def test_jax_block_parameter_preserves_graph_and_batching():
    jax = pytest.importorskip("jax", reason="JAX is optional")
    jnp = pytest.importorskip("jax.numpy")
    first = Vector(jnp.array([1.0, 2.0], dtype=jnp.float32))
    second = Scalar(jnp.array([[3.0], [4.0]], dtype=jnp.float32))
    parameter = BlockParameter([first, second])

    flattened = jax.jit(lambda: parameter.coordinate)()
    rebuilt = jax.jit(
        lambda coordinate: parameter.with_coordinate(coordinate).coordinate
    )(flattened + 1.0)
    gradient = jax.grad(
        lambda coordinate: jnp.sum(
            parameter.with_coordinate(coordinate).coordinate**2
        )
    )(flattened)

    np.testing.assert_array_equal(
        flattened,
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]]),
    )
    np.testing.assert_array_equal(rebuilt, np.asarray(flattened) + 1.0)
    np.testing.assert_array_equal(gradient, 2.0 * np.asarray(flattened))


def test_pytorch_block_parameter_preserves_graph_and_batching():
    torch = pytest.importorskip("torch", reason="PyTorch is optional")
    firstCoordinate = torch.tensor(
        [1.0, 2.0], dtype=torch.float64, requires_grad=True
    )
    secondCoordinate = torch.tensor(
        [[3.0], [4.0]], dtype=torch.float64, requires_grad=True
    )
    parameter = BlockParameter([
        Vector(firstCoordinate), Scalar(secondCoordinate)
    ])

    flattened = parameter.coordinate
    replacement = parameter.with_coordinate(flattened * 2.0)
    replacement.coordinate.sum().backward()

    assert flattened.shape == (2, 3)
    assert replacement.backend.name == "pytorch"
    assert replacement.backendMetadata.dtype == torch.float64
    torch.testing.assert_close(
        firstCoordinate.grad, torch.full_like(firstCoordinate, 4.0)
    )
    torch.testing.assert_close(
        secondCoordinate.grad, torch.full_like(secondCoordinate, 2.0)
    )
