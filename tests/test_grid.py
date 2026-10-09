import numpy as np
import pytest

from styne.utility.grid import Grid, UniformGrid


def test_grid_basic():
    pts = [np.array([0.0]), np.array([0.5]), np.array([1.0])]
    g = Grid(pts)
    assert g.dimension == 1
    assert len(g) == 3
    np.testing.assert_allclose(g[0], np.array([0.0]))
    np.testing.assert_allclose(g.to_array(), np.array([[0.0], [0.5], [1.0]]))


def test_uniformgrid_1d():
    ug = UniformGrid(0.0, 1.0, 5)
    assert ug.dimension == 1
    assert len(ug) == 5
    np.testing.assert_allclose(ug.to_array().ravel(), np.linspace(0.0, 1.0, 5))


def test_uniformgrid_2d_indexing_and_array():
    ug = UniformGrid((0.0, 1.0, 2), (10.0, 12.0, 3))
    assert ug.dimension == 2
    assert len(ug) == 6
    np.testing.assert_allclose(ug[1, 2], np.array([1.0, 12.0]))
    np.testing.assert_allclose(ug[3], np.array([1.0, 10.0]))
    arr = ug.to_array()
    assert arr.shape == (6, 2)
    expected = np.array([
        [0.0, 10.0],
        [0.0, 11.0],
        [0.0, 12.0],
        [1.0, 10.0],
        [1.0, 11.0],
        [1.0, 12.0],
    ])
    np.testing.assert_allclose(arr, expected)


def test_grid_preserves_pytorch_site_arrays():
    torch = pytest.importorskip("torch", reason="PyTorch is optional")
    points = torch.tensor(
        [[0.0, 0.5], [1.0, 1.5]], dtype=torch.float32, requires_grad=True
    )

    sites = Grid(points)
    stacked = sites.to_array()

    assert isinstance(stacked, torch.Tensor)
    assert stacked.dtype == torch.float32
    assert stacked.device == points.device
    stacked.sum().backward()
    torch.testing.assert_close(points.grad, torch.ones_like(points))


def test_grid_preserves_jax_site_arrays():
    jnp = pytest.importorskip("jax.numpy", reason="JAX is optional")
    points = jnp.array([[0.0], [0.5]], dtype=jnp.float32)

    stacked = Grid(points).to_array()

    assert type(stacked) is type(points)
    assert stacked.dtype == points.dtype
    np.testing.assert_allclose(stacked, points)
