"""
Extension point for custom forward maps.

To add a new forward map, subclass 'ForwardMap' and implement four things:

    pType   — the 'Parameter' subclass (or tuple of subclasses) your
               forward map accepts; used by 'prepare' for an isinstance
               check.
    pDim    — total number of scalar degrees of freedom.
    _prepare(parameter) — extract parameter state and perform all
               parameter-dependent precomputation (e.g. operator assembly,
               factorisations). Return everything '_evaluate' and the
               optional gradient hooks will need.
    _evaluate(preparedState) — compute and return the model response. For
               use with any built-in likelihood this must be a 1-D ndarray
               of length N (number of observations): the forward map
               evaluated at the observation sites. GLM-type response
               families read this vector as the linear predictor eta.

Callers drive the forward map through a two-phase protocol:

    preparedState = forwardMap.prepare(parameter)
    eta = forwardMap.evaluate(preparedState)

Gradient support (optional)
---------------------------
Implementing the 'DifferentiableForwardMap' protocol on your subclass unlocks
gradient-based MCMC (e.g. MALA). The library detects this at runtime via
isinstance; no registration required.
"""
from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Union, Any, runtime_checkable, Protocol
from styne.parameter.parameter import Parameter


class ForwardMap(ABC):
    """
    Base class for parametric forward maps: parameter -> model response
    at the observation sites.

    See the module docstring for the full extension guide. The base class
    holds no evaluation state. Subclasses return parameter-dependent state
    from 'prepare' and consume it in 'evaluate'. A subclass may still mutate
    a held child object during evaluation (for example a Gaussian process
    coordinate) as an interim measure; no caller may rely on that mutation
    persisting between calls.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def pType(self) -> Union[type, tuple]:
        """Parameter type accepted by this model.

        Return a single 'Parameter' subclass or a tuple of subclasses.
        Passed directly to 'isinstance' inside 'prepare'.
        """

    @property
    @abstractmethod
    def pDim(self) -> int:
        """Total number of scalar degrees of freedom in the parameter."""
        ...

    def prepare(self, parameter: Parameter) -> Any:
        """Return parameter-dependent state for a later evaluation."""
        if not isinstance(parameter, self.pType):
            raise TypeError(
                f"expected parameter of type {self.pType}, "
                f"got {type(parameter).__name__}"
            )

        return self._prepare(parameter)

    def evaluate(self, preparedState: Any) -> Any:
        """Evaluate and return the response for explicitly prepared state."""
        try:
            return self._evaluate(preparedState)

        except Exception as e:
            raise RuntimeError("ForwardMap evaluation failed") from e

    def __call__(self, parameter: Parameter) -> Any:
        """Prepare and evaluate a parameter in one call."""
        return self.evaluate(self.prepare(parameter))

    # ------------------------------------------------------------------
    # Protected subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _prepare(self, parameter: Parameter) -> Any:
        """Return all parameter-dependent state needed for evaluation."""
        ...

    @abstractmethod
    def _evaluate(self, preparedState: Any) -> Any:
        """Compute and return the response for 'preparedState'."""
        ...


@runtime_checkable
class DifferentiableForwardMap(Protocol):
    """
    Protocol for models that expose Jacobian actions.

    Implement this on a 'ForwardMap' subclass to enable gradient-based MCMC
    (e.g. MALA). The library detects support at runtime via
    'isinstance(model, DifferentiableForwardMap)'; no registration needed.

    Both methods receive their needed parameter state through their own
    arguments; they must not depend on a previous forward-map evaluation.
    """

    def directional_derivative(
            self, parameter: Parameter,
            direction: Parameter) -> ndarray:
        """
        Apply the model Jacobian J to a direction in parameter space.

        Parameters
        ----------
        parameter : Parameter
            Point at which the derivative is evaluated.
        direction : Parameter
            Direction vector in parameter space.

        Returns
        -------
        ndarray, shape (N,)
            D G(parameter)[direction], a vector in observation space.
        """
        ...

    def adjoint_derivative(
            self, parameter: Parameter,
            cotangent: ndarray) -> ndarray:
        """
        Apply the adjoint Jacobian J^T to a vector in observation space.

        Parameters
        ----------
        parameter : Parameter
            Point at which the derivative is evaluated.
        cotangent : ndarray, shape (N,)
            Vector in observation space (e.g. a score residual).

        Returns
        -------
        ndarray, shape (pDim,)
            J^T @ w, a vector in parameter space.
        """
        ...
