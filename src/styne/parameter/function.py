from __future__ import annotations

from styne.model.representation.expansion import Expansion
from styne.parameter.parameter import Parameter, _as_coordinate


class Function(Parameter):
    """Evaluable parameter binding coefficients to an ``Expansion``.

    For coefficients ``coordinate`` and expansion ``E``, ``evaluate(grid)``
    computes ``E.evaluate(coordinate, grid)``. Replacing the coordinate shares
    ``E`` because it contains representation data, not current parameter state.

    Parameters
    ----------
    coordinate : array-like
        Coefficients with shape ``(..., dimension)``. Leading dimensions are
        batch dimensions.
    expansion : Expansion
        Static mathematical representation evaluated by the coefficients.
    """

    def __init__(self, coordinate, expansion: Expansion):
        self._coordinate = self._validate(coordinate, expansion.dimension)
        self._expansion = expansion

    @property
    def dimension(self) -> int:
        return self._expansion.dimension

    @property
    def coordinate(self):
        return self._coordinate

    @property
    def expansion(self) -> Expansion:
        """Static evaluation strategy shared by reconstructed parameters."""
        return self._expansion

    def evaluate(self, grid):
        """Evaluate this parameter's coefficients on ``grid``."""
        return self._expansion.evaluate(self._coordinate, grid)

    def clone(self) -> Function:
        """Return an equivalent parameter sharing the static expansion."""
        backend = self.backend
        metadata = backend.metadata(self._coordinate)
        zero = backend.zeros(
            (), dtype=metadata.dtype, device=metadata.device
        )
        return self.with_coordinate(self._coordinate + zero)

    @staticmethod
    def _validate(coordinate, dimension):
        coordinate = _as_coordinate(coordinate)
        if coordinate.ndim < 1 or coordinate.shape[-1] != dimension:
            raise ValueError(
                "Function coordinate shape must be (..., dimension); "
                f"got {coordinate.shape} for dimension {dimension}."
            )
        return coordinate

    def with_coordinate(self, coordinate) -> Function:
        return self.__class__(coordinate, self._expansion)
