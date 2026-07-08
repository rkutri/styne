import numpy as np
from typing import List

from styne.statistics.interface import DensityInterface
from styne.parameter.parameter import Parameter


class LogScalingWrapper(DensityInterface):
    """
    Density scaled by a fixed log-factor, for tempering.

    Wraps another `DensityInterface` and multiplies its log-density (and
    log-gradient, if available) by a constant `scaling`. `scaling < 1`
    flattens the density, `scaling > 1` sharpens it.

    Parameters
    ----------
    density : DensityInterface
        The density to scale.
    scaling : float
        The multiplicative factor theta.
    """

    def __init__(self, density: DensityInterface, scaling: float):

        if not isinstance(density, DensityInterface):
            raise TypeError("density must adhere to density interface.")

        self._baseDensity = density
        self._theta = scaling

    @property
    def domainType(self) -> type:
        return self._baseDensity.domainType

    @property
    def domainDimension(self) -> int:
        return self._baseDensity.domainDimension

    @property
    def baseDensity(self) -> DensityInterface:
        return self._baseDensity

    @property
    def scaling(self):
        return self._theta

    def evaluate_log(self, state: Parameter) -> float:
        """
        `scaling * baseDensity.evaluate_log(state)`.

        Parameters
        ----------
        state : Parameter

        Returns
        -------
        float
        """
        return self._theta * self._baseDensity.evaluate_log(state)

    def evaluate_log_gradient(self, state: Parameter) -> np.ndarray:
        """
        `scaling * baseDensity.evaluate_log_gradient(state)`.

        Parameters
        ----------
        state : Parameter

        Returns
        -------
        np.ndarray
        """
        return self._theta * self._baseDensity.evaluate_log_gradient(state)


class ProductWrapper(DensityInterface):
    """
    Product of several densities, as a sum of their log-densities.

    All factors must share the same domain dimension. Domain type
    compatibility is checked at construction, allowing for factors whose
    `domainType` is itself a tuple of acceptable types, in which case the
    wrapper's own `domainType` narrows to the intersection across factors.

    Parameters
    ----------
    factors : List[DensityInterface]
        At least one density, all with matching domain dimension and
        compatible domain type.
    """

    def __init__(self, factors: List[DensityInterface]):

        if len(factors) == 0:
            raise ValueError("ProductWrapper requires at least one factor.")

        self._domainType = factors[0].domainType
        self._domainDimension = factors[0].domainDimension

        for factor in factors:
            if not isinstance(factor, DensityInterface):
                raise TypeError("All factors must adhere to DensityInterface.")
            if factor.domainDimension != self._domainDimension:
                raise ValueError("All factors must have the same domain dimension.")

            # Check domain type compatibility, allowing for tuples (OR types)
            t1 = self._domainType
            t2 = factor.domainType
            if isinstance(t1, tuple) and isinstance(t2, tuple):
                # Intersection must not be empty
                common = set(t1).intersection(set(t2))
                if not common:
                    raise ValueError("Incompatible domain types (no intersection).")
                self._domainType = tuple(common) if len(common) > 1 else list(common)[0]
            elif isinstance(t1, tuple):
                if t2 not in t1:
                    raise ValueError(f"Domain type {t2} not in accepted {t1}.")
                self._domainType = t2
            elif isinstance(t2, tuple):
                if t1 not in t2:
                    raise ValueError(f"Domain type {t1} not in accepted {t2}.")
                # keep t1
            elif t1 != t2:
                raise ValueError(f"All factors must have the same domain type. {t1} != {t2}")

        self._factors = list(factors)

    @property
    def domainType(self) -> type:
        return self._domainType

    @property
    def domainDimension(self) -> int:
        return self._domainDimension

    def factor(self, idx: int) -> DensityInterface:
        """
        The factor at index `idx`.

        Parameters
        ----------
        idx : int

        Returns
        -------
        DensityInterface
        """
        return self._factors[idx]

    def evaluate_log(self, state: Parameter) -> float:
        """
        Sum of every factor's `evaluate_log(state)`.

        Parameters
        ----------
        state : Parameter

        Returns
        -------
        float
        """
        return sum(f.evaluate_log(state) for f in self._factors)

    def evaluate_log_gradient(self, state: Parameter) -> np.ndarray:
        """
        Sum of every factor's `evaluate_log_gradient(state)`.

        Parameters
        ----------
        state : Parameter

        Returns
        -------
        np.ndarray
        """
        return sum(f.evaluate_log_gradient(state) for f in self._factors)