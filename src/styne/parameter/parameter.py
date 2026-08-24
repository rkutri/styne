"""Base class for immutable model parameters.

A Parameter wraps the backend-native coordinate array that the model and MCMC
layer operate on. State transitions construct replacements through
``with_coordinate`` rather than replacing coordinates in place.

To define a custom parameter, subclass ``Parameter`` and implement
``dimension``, a read-only ``coordinate`` property, and ``with_coordinate``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from styne.backend import BackendInferenceError, get_backend, infer_backend


def _as_coordinate(coordinate):
    """Preserve native arrays and default non-array input to NumPy."""
    try:
        backend = infer_backend(coordinate)
    except BackendInferenceError:
        backend = get_backend("numpy")
        coordinate = backend.asarray(coordinate)

    if coordinate.ndim == 0:
        coordinate = backend.namespace.expand_dims(coordinate, axis=0)

    return coordinate


class Parameter(ABC):
    """Finite-dimensional parameter with a read-only coordinate array.

    Notes
    -----
    The coordinate remains native to its numerical backend. Subclasses may
    carry static metadata for the forward model alongside that array.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of scalar degrees of freedom."""
        pass

    @property
    @abstractmethod
    def coordinate(self):
        """Backend-native coordinate array representing the parameter."""
        pass

    @abstractmethod
    def with_coordinate(self, coordinate) -> Parameter:
        """Return a parameter with ``coordinate`` without changing this one."""
        pass

    @property
    def backend(self):
        """Numerical backend inferred from the coordinate array."""
        return infer_backend(self.coordinate)

    @property
    def backendMetadata(self):
        """Dtype and device metadata computed from the coordinate array."""
        return self.backend.metadata(self.coordinate)
