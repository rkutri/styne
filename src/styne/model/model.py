"""
Extension point for custom forward models.

To add a new model, subclass 'Model' and implement four things:

    pType   — the 'Parameter' subclass (or tuple of subclasses) your model
               accepts; used by 'interpolate' for an isinstance check.
    pDim    — total number of scalar degrees of freedom.
    _interpolate(parameter) — extract and cache whatever internal state
               '_evaluate' will need from 'parameter'. Do not touch
               'self._evaluation' here.
    _evaluate() — compute the model response and assign the result to
               'self._evaluation'. For use with any built-in likelihood
               this must be a 1-D ndarray of length N (number of
               observations), representing the linear predictor eta.

Callers drive the model through a two-phase protocol:

    model.interpolate(parameter)   # phase 1: configure
    model.evaluate()               # phase 2: compute (cached)
    eta = model.evaluation         # read result

Gradient support (optional)
---------------------------
Implementing the 'DifferentiableModel' protocol on your subclass unlocks
gradient-based MCMC (e.g. MALA). The library detects this at runtime via
isinstance; no registration required.
"""
from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Union, Any, runtime_checkable, Protocol
from styne.parameter.parameter import Parameter


class Model(ABC):
    """
    Base class for parametric forward models.

    See the module docstring for the full extension guide. Instances are
    stateful and not thread-safe; one model instance should be owned by
    exactly one likelihood object.
    """

    def __init__(self):
        self._evaluation = None
        self._interpolatedParameter = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def evaluation(self) -> Any:
        """Last successful evaluation result (the linear predictor eta).

        A 1-D ndarray of length N set by '_evaluate'. None if the model
        has not yet been evaluated or has been reset.
        """
        return self._evaluation

    @property
    @abstractmethod
    def pType(self) -> Union[type, tuple]:
        """Parameter type accepted by this model.

        Return a single 'Parameter' subclass or a tuple of subclasses.
        Passed directly to 'isinstance' inside 'interpolate'.
        """

    @property
    @abstractmethod
    def pDim(self) -> int:
        """Total number of scalar degrees of freedom in the parameter."""
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
        """Extract and cache internal state from 'parameter'.

        Store whatever '_evaluate' will need (e.g. coordinate arrays,
        GP coefficients). Called once per distinct parameter object;
        repeated calls with the same object are short-circuited by
        the base class. Must not write to 'self._evaluation'.
        """
        ...

    @abstractmethod
    def _evaluate(self) -> None:
        """Compute the model response and assign it to 'self._evaluation'.

        Must assign a 1-D ndarray of shape (N,) to 'self._evaluation',
        where N is the number of observation sites. This value is read
        by the likelihood layer as the linear predictor eta.
        """
        ...


@runtime_checkable
class DifferentiableModel(Protocol):
    """
    Protocol for models that expose Jacobian actions.

    Implement this on a 'Model' subclass to enable gradient-based MCMC
    (e.g. MALA). The library detects support at runtime via
    'isinstance(model, DifferentiableModel)'; no registration needed.

    Both methods are called after 'interpolate' and 'evaluate' have run,
    so internal model state is guaranteed to be current.
    """

    def directional_derivative(self, parameter: Parameter) -> ndarray:
        """
        Apply the model Jacobian J to a direction in parameter space.

        Parameters
        ----------
        parameter : Parameter
            Direction vector in parameter space.

        Returns
        -------
        ndarray, shape (N,)
            J @ parameter.coordinate, a vector in observation space.
        """
        ...

    def adjoint_directional_derivative(self, w: ndarray) -> ndarray:
        """
        Apply the adjoint Jacobian J^T to a vector in observation space.

        Parameters
        ----------
        w : ndarray, shape (N,)
            Vector in observation space (e.g. a score residual).

        Returns
        -------
        ndarray, shape (pDim,)
            J^T @ w, a vector in parameter space.
        """
        ...