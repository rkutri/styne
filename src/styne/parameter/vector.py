from __future__ import annotations

from numpy import ndarray, asarray, atleast_1d

from styne.parameter.parameter import Parameter


class Vector(Parameter):
    """
    Finite-dimensional parameter vector.
    """

    def __init__(self, coordinate: ndarray):
        self._coordinate = self.validate(coordinate)

    @property
    def dimension(self) -> int:
        return self._coordinate.size

    @property
    def coordinate(self) -> ndarray:
        return self._coordinate

    @coordinate.setter
    def coordinate(self, coordinate: ndarray) -> None:
        self._coordinate = self.validate(coordinate)

    @staticmethod
    def validate(coordinate: ndarray):

        coordinate = atleast_1d(asarray(coordinate, dtype=float))

        if coordinate.ndim not in [1, 2]:
            raise ValueError(
                "Parameter coordinates must be 1D vector or 2D batch matrix."
            )

        return coordinate

    def clone(self) -> Parameter:
        return self.__class__(self._coordinate.copy())
