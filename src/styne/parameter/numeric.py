from __future__ import annotations

import numpy as np
from styne.parameter.parameter import Parameter


class Numeric(Parameter):
    """
    Minimal wrapper around a NumPy array
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
        self._array = value

    def clone(self) -> Numeric:
        return Numeric(self._array.copy())
