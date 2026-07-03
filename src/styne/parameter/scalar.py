from __future__ import annotations

import numpy as np
from math import isclose

from styne.parameter.parameter import Parameter


class Scalar(Parameter):
    """
    Scalar-valued parameter, a single float wrapped as a length-1
    coordinate array.
    """

    def __init__(self, coordinate):
        self._coordinate = self._as_scalar_array(coordinate)

    def _as_scalar_array(self, value) -> np.ndarray:

        if isinstance(value, np.ndarray):
            arr = np.atleast_1d(value)

        else:
            arr = np.atleast_1d(value)

        if arr.ndim != 1 or arr.size != 1:
            raise ValueError(
                f"Scalar requires a 1D numpy array of length 1. "
                f"Got shape={arr.shape}"
            )
        return arr.astype(float)

    @property
    def dimension(self):
        """
        Always 1.
        """
        return 1

    @property
    def coordinate(self) -> np.ndarray:
        """
        The scalar value as a length-1 array.
        """
        return self._coordinate

    @coordinate.setter
    def coordinate(self, value):
        self._coordinate = self._as_scalar_array(value)

    def __eq__(self, other: Scalar):

        if isinstance(other, Scalar):
            return isclose(float(self._coordinate[0]),
                           float(other._coordinate[0]))

        return NotImplemented

    def clone(self):
        """
        Return an independent copy with the same value.

        Returns
        -------
        Scalar
        """
        return Scalar(self._coordinate.copy())
