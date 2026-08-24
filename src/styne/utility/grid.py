"""
Thin wrappers around a list of numpy point arrays. Mainly
in order to be able to discern between sets of points and
uniform grids.
"""
import numpy as np

from typing import Iterable, Iterator

from styne.backend import BackendInferenceError, infer_backend


def _as_array(value):
    """Preserve registered backend arrays; convert Python values to NumPy."""
    try:
        infer_backend(value)
    except BackendInferenceError:
        return np.asarray(value)
    return value


def _copy_array(array):
    if hasattr(array, "copy"):
        return array.copy()
    if hasattr(array, "clone"):
        return array.clone()
    return array


class Grid:
    """
    Thin wrapper around a list of point arrays, letting a set of arbitrary
    points and a `UniformGrid` be used interchangeably wherever a `Grid` is
    expected.

    Parameters
    ----------
    points : iterable of array-like
        Points, each reshaped to a flat array. Registered backend arrays are
        retained without conversion. All points must have the same dimension.
    """

    def __init__(self, points: Iterable):

        self._points = [_as_array(p).reshape(-1) for p in points]

        if len(self._points) == 0:
            self._dimension = 0

        else:

            d = int(self._points[0].shape[0])
            for p in self._points:
                if int(p.shape[0]) != d:
                    raise ValueError("all points must have the same dimension")

            self._dimension = d

    @property
    def dimension(self) -> int:
        return self._dimension

    def __len__(self) -> int:
        return len(self._points)

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            return self.to_array()[idx]

        if isinstance(idx, int):
            return _copy_array(self._points[idx])

        raise TypeError("index must be int or slice")

    def __iter__(self) -> Iterator:
        for p in self._points:
            yield _copy_array(p)

    def to_array(self):
        """
        Stack all points into a single `(n_points, dimension)` array.

        Returns
        -------
        array-like
        """

        if len(self._points) == 0:
            return np.empty((0, 0))

        backend = infer_backend(self._points[0])
        return backend.namespace.stack(self._points, axis=0)

    def __repr__(self) -> str:
        return f"Grid(n_points={len(self)}, dimension={self.dimension})"


class UniformGrid(Grid):
    """Uniform grid. 1D wraps `np.linspace`, 2D stores two axes.

    Constructor signatures:
    - `UniformGrid(start, stop, num)` for 1D
    - `UniformGrid((x0, x1, nx), (y0, y1, ny))` for 2D
    """

    def __init__(self, *args):
        
        # 1D
        if len(args) == 3 and isinstance(args[2], int):

            start, stop, num = args
            self._xAxis = np.linspace(float(start), float(stop), int(num), dtype=float)
            self._is2d = False
            self._nX = self._xAxis.size
            self._dimension = 1

            return

        # 2D
        if len(args) == 2 and all(isinstance(a, (list, tuple)) and len(a) == 3 for a in args):

            (x0, x1, nx), (y0, y1, ny) = args
            self._xAxis = np.linspace(float(x0), float(x1), int(nx), dtype=float)
            self._yAxis = np.linspace(float(y0), float(y1), int(ny), dtype=float)
            self._is2d = True
            self._nX = self._xAxis.size
            self._nY = self._yAxis.size
            self._dimension = 2

            return

        raise TypeError(
            "UniformGrid expects (start, stop, num) for 1D or ((x0,x1,nx),(y0,y1,ny)) for 2D"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def __len__(self) -> int:
        return int(self._nX if not self._is2d else self._nX * self._nY)

    def __getitem__(self, idx):

        if self._is2d:

            # tuple indexing (ix, iy)
            if isinstance(idx, tuple):

                if len(idx) != 2:
                    raise IndexError("tuple index must be length 2")

                ix, iy = idx
                return np.array([self._xAxis[int(ix)], self._yAxis[int(iy)]], dtype=float)

            # flattened integer indexing
            if isinstance(idx, int):
                
                n = int(idx)
                if n < 0:
                    n += len(self)

                if n < 0 or n >= len(self):
                    raise IndexError("index out of range")

                ix = n // self._nY
                iy = n % self._nY

                return np.array([self._xAxis[ix], self._yAxis[iy]], dtype=float)

            if isinstance(idx, slice):
                return self.to_array()[idx]

            raise TypeError("index must be int, slice, or tuple")

        # 1D
        if isinstance(idx, int):

            n = int(idx)
            if n < 0:
                n += len(self)
            return np.array([self._xAxis[n]], dtype=float)

        if isinstance(idx, slice):
            return self.to_array()[idx]
        raise TypeError("index must be int or slice for 1D UniformGrid")

    def __iter__(self) -> Iterator[np.ndarray]:
        for row in self.to_array():
            yield row

    @property
    def axis(self) -> np.ndarray:
        if self._is2d:
            raise AttributeError(
                "axis is only available for 1D UniformGrid; use xAxis/yAxis")
        return self._xAxis.copy()

    @property
    def xAxis(self) -> np.ndarray:
        return self._xAxis.copy()

    @property
    def yAxis(self) -> np.ndarray:
        if not self._is2d:
            raise AttributeError(
                "yAxis is only available for 2D UniformGrid")
        return self._yAxis.copy()

    def to_array(self) -> np.ndarray:
        """
        Stack the grid into a `(n_points, dimension)` array.

        For 2D, returns every `(x, y)` combination in row-major order,
        `node(i, j) = i * nY + j`, matching `__getitem__`'s flattened indexing.

        Returns
        -------
        np.ndarray
        """

        if self._is2d:

            xs = np.repeat(self._xAxis, self._nY)
            ys = np.tile(self._yAxis, self._nX)

            return np.column_stack((xs, ys))

        return self._xAxis.reshape(-1, 1).copy()
