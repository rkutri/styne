from __future__ import annotations

from styne.parameter.parameter import Parameter, _as_coordinate


class Vector(Parameter):
    """Finite-dimensional vector or batch of vectors."""

    def __init__(self, coordinate):
        self._coordinate = self._validate(coordinate)

    @property
    def dimension(self) -> int:
        return self._coordinate.shape[-1]

    @property
    def coordinate(self):
        return self._coordinate

    @staticmethod
    def _validate(coordinate):
        coordinate = _as_coordinate(coordinate)
        if coordinate.ndim not in (1, 2):
            raise ValueError(
                "Vector coordinate shape must be (dimension,) or "
                "(batch, dimension)."
            )
        return coordinate

    def with_coordinate(self, coordinate) -> Vector:
        return self.__class__(coordinate)
