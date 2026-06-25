from __future__ import annotations

import numpy as np

from styne.model.representation.expansion import Expansion, GridFunctionInterface
from styne.parameter.parameter import Parameter


class Function(Parameter):

    def __init__(self, expansion: Expansion):
        self._expansion = expansion

    @property
    def dimension(self) -> int:
        return self._expansion.dimension

    @property
    def coordinate(self) -> np.ndarray:
        return self._expansion.coefficient

    @coordinate.setter
    def coordinate(self, value: np.ndarray) -> None:
        self._expansion.coefficient = value

    @property
    def function(self) -> GridFunctionInterface:
        return self._expansion

    def clone(self) -> Function:
        return self.__class__(self._expansion.clone())
