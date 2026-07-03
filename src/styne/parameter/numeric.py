from __future__ import annotations

import numpy as np
from styne.parameter.parameter import Parameter


class Numeric(Parameter):
    """
    Minimal wrapper around a NumPy array as a parameter.

    The coordinate setter coerces to a float array and requires the shape to
    match whatever's currently stored, raising ValueError on a mismatch.
    Shape is effectively fixed once constructed. The constructor itself does
    not coerce or validate, whatever's passed to `array` is stored as-is, so
    `dimension` and `coordinate` will fail downstream, not at construction,
    if it isn't already an ndarray.

    Parameters
    ----------
    array : np.ndarray
    """

    def __init__(self, array: np.ndarray):

        super().__init__()
        self._array = array

    @property
    def dimension(self) -> int:
        return self._array.size

    @property
    def coordinate(self) -> np.ndarray:
        return self._array

    @coordinate.setter
    def coordinate(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float)
        if value.shape != self._array.shape:
            raise ValueError(
                f"Shape mismatch: expected {self._array.shape}, "
                f"got {value.shape}."
            )
        self._array = value

    def clone(self) -> Numeric:
        """
        Return an independent copy with the same array.

        Returns
        -------
        Numeric
        """
        return Numeric(self._array.copy())
