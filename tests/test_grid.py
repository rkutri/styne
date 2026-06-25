import numpy as np

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
