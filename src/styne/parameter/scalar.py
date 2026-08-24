from __future__ import annotations

from styne.parameter.parameter import Parameter, _as_coordinate


class Scalar(Parameter):
    """Scalar parameter or batch with a trailing dimension of one."""

    def __init__(self, coordinate):
        self._coordinate = self._validate(coordinate)

    @staticmethod
    def _validate(coordinate):
        coordinate = _as_coordinate(coordinate)
        if coordinate.ndim not in (1, 2) or coordinate.shape[-1] != 1:
            raise ValueError(
                "Scalar coordinate shape must be (1,) or (batch, 1); "
                f"got {coordinate.shape}."
            )
        return coordinate

    @property
    def dimension(self):
        return 1

    @property
    def coordinate(self):
        return self._coordinate

    def clone(self):
        """Return an equivalent parameter with independent array storage."""
        backend = self.backend
        metadata = backend.metadata(self._coordinate)
        zero = backend.zeros(
            (), dtype=metadata.dtype, device=metadata.device
        )
        return self.with_coordinate(self._coordinate + zero)

    def with_coordinate(self, coordinate) -> Scalar:
        return self.__class__(coordinate)
