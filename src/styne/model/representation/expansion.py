from abc import ABC, abstractmethod
from typing import Protocol

from styne.utility.grid import Grid


class GridFunction(Protocol):
    """Structural contract for an object evaluable on a grid."""

    def evaluate(self, grid: Grid): ...


class Expansion(ABC):
    """Static strategy for evaluating finite-dimensional coefficients.

    Coefficients use a trailing feature axis, ``(..., dimension)``. Evaluation
    preserves all leading batch dimensions and replaces the feature axis with
    the expansion's output axis.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of coefficients accepted by the representation."""
        pass

    @abstractmethod
    def evaluate(self, coefficient, grid: Grid):
        """Evaluate coefficients shaped ``(..., dimension)`` on ``grid``."""
        pass
