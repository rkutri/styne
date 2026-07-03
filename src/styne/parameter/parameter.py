"""
Base class for model parameters.

A Parameter wraps the coordinate array that the model and MCMC layer
operate on. The library reads 'coordinate' to evaluate models, and
writes to 'coordinate' to propose new states. The rest of the library
never needs to know the semantic meaning of the coordinates; that
knowledge lives inside the Parameter subclass and the Model that
consumes it.

To define a custom parameter, subclass 'Parameter' and implement:

    dimension   - int, the number of scalar degrees of freedom.
    coordinate  - ndarray property with a getter *and* a setter.
                  The getter returns the current coordinate array;
                  the setter accepts an ndarray of the same shape
                  and updates internal state accordingly.
    clone()     - return an independent copy with identical content.

Built-in subclasses: 'Vector', 'Scalar', 'Function', 'BlockParameter'.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from numpy import ndarray, array_equal


class Parameter(ABC):
    """Finite-dimensional parameter with a mutable coordinate array.

    See the module docstring for the extension guide. Instances are
    unhashable; equality is strict elementwise identity.

    Notes
    -----
    The coordinate is always a NumPy ndarray. The reason for wrapping
    it in a Parameter object rather than passing the array directly is
    that subclasses can carry metadata for the forward model alongside
    the coordinates. For instance, a 'Function' parameter holds the basis
    expansion that lets the model evaluate the function on a grid, and
    a 'BlockParameter' preserves the blocking structure across
    sub-parameters.
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
        """Coordinate array representing the parameter.

        Subclasses must also provide a setter that accepts an ndarray
        of the same shape. The MCMC layer writes to this property to
        update the parameter state in-place.
        """
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
    def clone(self) -> Parameter:
        """Return an independent copy with identical content.

        The clone must not share mutable state with the original;
        writing to one's coordinate must not affect the other.
        """
        pass
