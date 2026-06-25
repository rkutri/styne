import numpy as np

from typing import Callable

from styne.model.representation.expansion import GridFunctionInterface
from styne.utility.grid import Grid


class ExplicitFunction(GridFunctionInterface):

    def __init__(self, fcn_callable):

        if not callable(fcn_callable):
            raise TypeError(
                "fcn_callable in explicit function must be callable."
            )

        self._fcn = fcn_callable

    def evaluate(self, grid: Grid) -> np.ndarray:
        return np.array([self._fcn(x) for x in grid])


class TransformWrapper(GridFunctionInterface):

    def __init__(self, operation: Callable,
                 gridFunction: GridFunctionInterface):
        self._op = operation
        self._gf = gridFunction

    def evaluate(self, grid: Grid) -> np.ndarray:
        return self._op(self._gf.evaluate(grid))
