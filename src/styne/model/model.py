from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Type, Any, runtime_checkable, Protocol
from styne.parameter.parameter import Parameter


class Model(ABC):
    """
    Template class for parametric models. For a given parameter,
    a model implementation should be able to provide the model response,
    such as forward-maps, or predictors and store it in self._evaluation.
    """

    def __init__(self):
        self._evaluation = None
        self._interpolatedParameter = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def evaluation(self) -> Any:
        """Return the last successful evaluation result."""
        return self._evaluation

    @property
    @abstractmethod
    def pType(self) -> Type[Parameter]:
        """Expected parameter type."""
        ...

    @property
    @abstractmethod
    def pDim(self) -> int:
        """Dimensionality of the parameter space."""
        ...

    def interpolate(self, parameter: Parameter) -> None:
        """
        Configure the model with a parameter.
        Clears the previous evaluation result.
        """
        if not isinstance(parameter, self.pType):
            raise TypeError("Parameter must match pType")

        if self._interpolatedParameter is parameter:
            return

        self._evaluation = None
        self._interpolate(parameter)
        self._interpolatedParameter = parameter

    def evaluate(self) -> None:
        """
        Evaluate the model-response using previously supplied parameter.
        Idempotent: skips if a valid evaluation already exists.
        """
        if self._evaluation is not None:
            return

        try:
            self._evaluate()

        except Exception as e:

            self._evaluation = None
            raise RuntimeError("Model evaluation failed") from e

    def reset(self) -> None:
        """Reset tracked state, forcing re-interpolation and
        re-evaluation on the next call."""
        self._evaluation = None
        self._interpolatedParameter = None

    # ------------------------------------------------------------------
    # Protected subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _interpolate(self, parameter: Parameter) -> None:
        """Implementation of interpolation."""
        ...

    @abstractmethod
    def _evaluate(self) -> None:
        """Implementation of model response."""
        ...


@runtime_checkable
class DifferentiableModel(Protocol):
    """
    Protocol for models that can compute directional derivatives.
    """

    def directional_derivative(self, parameter: Parameter) -> Parameter:
        """
        Apply the jacobian of the model response with respect to the parameters,
        to a given parameter.
        """
        ...

    def adjoint_directional_derivative(self, w: ndarray) -> ndarray:
        """
        Apply the adjoint (transpose) of the model Jacobian to w.
        Maps from observation space to parameter space.
        """
        ...