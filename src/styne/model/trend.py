from typing import Protocol, runtime_checkable

import numpy as np

from styne.utility.grid import Grid


@runtime_checkable
class Trend(Protocol):
    """Deterministic additive contribution to a SGLMM linear predictor."""

    def evaluate(self, sites: Grid) -> np.ndarray:
        """
        Evaluate the trend at a set of spatial sites.

        Parameters
        ----------
        sites : Grid

        Returns
        -------
        np.ndarray
        """
        ...


class ConstantTrend:
    """Uniform scalar offset at every observation site."""

    def __init__(self, offset: float):
        self._offset = float(offset)

    def evaluate(self, sites: Grid) -> np.ndarray:
        """
        Return the constant offset, broadcast to the number of sites.

        Parameters
        ----------
        sites : Grid

        Returns
        -------
        np.ndarray
        """
        return np.full(len(sites), self._offset)


class LinearTrend:
    """
    Inner product of fixed coefficients with spatial coordinates.

    coefficients must have length equal to the spatial dimension of sites.
    """

    def __init__(self, coefficients: np.ndarray):
        self._coefficients = np.asarray(coefficients, dtype=float).ravel()

    def evaluate(self, sites: Grid) -> np.ndarray:
        """
        Evaluate the linear trend, `coefficients @ site` at each site.

        Parameters
        ----------
        sites : Grid

        Returns
        -------
        np.ndarray
        """
        coordinates = sites.to_array()

        if coordinates.shape[1] != self._coefficients.size:
            raise ValueError(
                f"coefficients length {self._coefficients.size} does not "
                f"match site dimension {coordinates.shape[1]}"
            )

        return coordinates @ self._coefficients
