from __future__ import annotations

import numpy as np

from styne.model.representation.expansion import Expansion, GridFunctionInterface
from styne.parameter.parameter import Parameter


class Function(Parameter):
    """
    Function-valued parameter, backed by an `Expansion`.

    The coordinate is the expansion's coefficient array. Setting it
    delegates to the expansion's own `coefficient` setter, so any
    validation happens there, not in this class.

    Parameters
    ----------
    expansion : Expansion
        The underlying basis and coefficient store.
    """

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
        """
        The underlying `Expansion`, exposing basis-specific operations
        (`evaluate`, `evaluate_native`) beyond the flat coordinate view.
        """
        return self._expansion

    def clone(self) -> Function:
        """
        Return an independent copy, cloning the underlying expansion.

        Uses `self.__class__` rather than `Function` directly, so a subclass
        constructed the same way clones as its own type.

        Returns
        -------
        Function
        """
        return self.__class__(self._expansion.clone())
