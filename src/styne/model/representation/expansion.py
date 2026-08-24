from abc import ABC, abstractmethod
from typing import Protocol

from styne.backend import BackendCapabilityError, infer_backend
from styne.utility.grid import Grid


def backend_constant(value, reference):
    """Place static representation data beside a backend-native array."""
    backend = infer_backend(reference)
    metadata = backend.metadata(reference)
    return backend.asarray(
        value, dtype=metadata.dtype, device=metadata.device
    )


class GridFunction(Protocol):
    """Structural contract for an object evaluable on a grid."""

    def evaluate(self, grid: Grid): ...


class BoundExpansion(ABC):
    """An expansion bound to a fixed evaluation grid.

    Derivatives are evaluated at the supplied coefficient, so this contract
    supports nonlinear expansions. JAX and PyTorch obtain derivative actions
    from their native automatic-differentiation implementations. NumPy
    subclasses may provide analytical fallbacks.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def evaluate(self, coefficient):
        ...

    def directional_derivative(self, coefficient, direction):
        backend = infer_backend(coefficient, direction)
        if backend.capabilities.automaticDifferentiation:
            _, derivative = backend.jvp(
                self.evaluate, (coefficient,), (direction,)
            )
            return derivative
        return self._directional_derivative(coefficient, direction)

    def adjoint_derivative(self, coefficient, cotangent):
        backend = infer_backend(coefficient, cotangent)
        if backend.capabilities.automaticDifferentiation:
            _, pullback = backend.vjp(self.evaluate, coefficient)
            result = pullback(cotangent)
            return result[0] if isinstance(result, tuple) else result
        return self._adjoint_derivative(coefficient, cotangent)

    def _directional_derivative(self, coefficient, direction):
        raise BackendCapabilityError(
            f"{type(self).__name__} has no NumPy directional derivative."
        )

    def _adjoint_derivative(self, coefficient, cotangent):
        raise BackendCapabilityError(
            f"{type(self).__name__} has no NumPy adjoint derivative."
        )


class BoundLinearExpansion(BoundExpansion):
    """Grid-bound evaluation known to be linear in its coefficient."""

    def _directional_derivative(self, coefficient, direction):
        return self.evaluate(direction)


class Expansion(ABC):
    """Static strategy for evaluating finite-dimensional coefficients.

    Coefficients use a trailing feature axis, ``(..., dimension)``. Evaluation
    preserves all leading batch dimensions and replaces the feature axis with
    the expansion's output axis. No linearity is assumed.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of coefficients accepted by the representation."""
        pass

    def bind(self, grid: Grid) -> BoundExpansion:
        """Bind and cache static grid-dependent evaluation data."""
        if not hasattr(self, "_boundEvaluations"):
            self._boundEvaluations = {}
        key = id(grid)
        cached = self._boundEvaluations.get(key)
        if cached is None or cached[0] is not grid:
            cached = (grid, self._bind(grid))
            self._boundEvaluations[key] = cached
        return cached[1]

    @abstractmethod
    def _bind(self, grid: Grid) -> BoundExpansion:
        """Construct the evaluator used by `bind`."""
        ...

    def evaluate(self, coefficient, grid: Grid):
        """Evaluate coefficients shaped ``(..., dimension)`` on ``grid``."""
        return self.bind(grid).evaluate(coefficient)


class LinearExpansion(Expansion):
    """Expansion whose evaluation is linear in its coefficient."""

    @abstractmethod
    def _bind(self, grid: Grid) -> BoundLinearExpansion:
        ...
