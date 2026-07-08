from __future__ import annotations

from numpy import ndarray, asarray, atleast_1d

from styne.parameter.parameter import Parameter


class Vector(Parameter):
    """
    Finite-dimensional parameter vector.

    Parameters
    ----------
    coordinate : ndarray
        1D vector or 2D batch matrix, coerced to float and validated on
        every set, not just at construction.
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
        """
        Coerce to a float array and check it's 1D or 2D.

        Parameters
        ----------
        coordinate : ndarray

        Returns
        -------
        ndarray

        Raises
        ------
        ValueError
            If the array isn't 1D or 2D after coercion.
        """

        coordinate = atleast_1d(asarray(coordinate, dtype=float))

        if coordinate.ndim not in [1, 2]:
            raise ValueError(
                "Parameter coordinates must be 1D vector or 2D batch matrix."
            )

        return coordinate

    def clone(self) -> Parameter:
        """
        Return an independent copy with the same coordinate.

        Returns
        -------
        Vector
        """
        return self.__class__(self._coordinate.copy())
