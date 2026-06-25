from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from numpy import ndarray, array_equal


class Parameter(ABC):
    """
    Template class for parameters represented by finite-dimensional real
    coordinates. Coordinates are NumPy arrays. Equality is strict elementwise
    identity; instances are unhashable.

    Design 
    ------
    Conceptually, the Parameter class acts as a bridge between the abstract 
    object the model parameters are supposed to represent, and the NumPy
    array representing the coordinate array. Ultimately, all inference and
    sampling is performed with respect to this coordinate array. It's the
    interface between the model and the rest of the library.
    """

    __hash__ = None  # equality defined, hashing disabled

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of scalar degrees of freedom."""
        pass

    @property
    @abstractmethod
    def coordinate(self) -> ndarray:
        """Coordinate vector representing the parameter."""
        pass

    def __eq__(self, other: object) -> bool:
        """
        Parameters are equal if they have the same dimension and identical
        coordinate entries. NaNs compare unequal.
        """
        if not isinstance(other, Parameter):
            return NotImplemented

        if self.dimension != other.dimension:
            return False

        # Exact match in all entries
        return array_equal(self.coordinate, other.coordinate)

    @abstractmethod
    def clone(self, memo: Optional[dict[int, Any]] = None) -> Parameter:
        """
        Return an independent parameter with identical content. May delegate
        to deepcopy or use a faster method.
        """
        pass
