from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, runtime_checkable, Protocol

from numpy import ndarray
from numpy.random import Generator, SeedSequence

from styne.model.forwardmap import ForwardMap
from styne.parameter.parameter import Parameter
from styne.statistics.measure import ProbabilityMeasure
from styne.statistics.data import Data


SeedType = Union[int, Sequence[int], SeedSequence, None]


class DensityInterface(ABC):
    """
    Interface for objects exposing a log-density over a `Parameter` domain.

    Parameters
    ----------
    (abstract, no constructor)

    Notes
    -----
    Subclasses implement `domainType`, `domainDimension`, and `evaluate_log`.
    The interface makes no guarantee about normalisation. `evaluate_log` may
    return a properly normalised log-density or one missing an additive
    constant, depending on the implementation. Check the concrete class's
    own docstring, don't assume either way. This matters for MCMC code that
    compares densities across different classes rather than only within one.
    """

    @property
    @abstractmethod
    def domainType(self) -> Parameter:
        ...

    @property
    @abstractmethod
    def domainDimension(self) -> int:
        ...

    @abstractmethod
    def evaluate_log(self, state: Parameter) -> float:
        ...


@runtime_checkable
class DifferentiableDensity(Protocol):
    """
    Protocol for densities that expose a log-gradient.
    """

    def evaluate_log_gradient(self, state: Parameter) -> ndarray:
        """
        Evaluate the gradient of the logarithm of the density.
        """

        ...


@runtime_checkable
class TwiceDifferentiableDensity(DifferentiableDensity, Protocol):
    """
    Protocol for densities that expose a log-hessian.
    """

    def evaluate_log_hessian(self, state: Parameter) -> ndarray:
        """
        Evaluate the Hessian of the logarithm of the density.
        """

        ...


class LikelihoodInterface(DensityInterface):
    """
    Density interface for likelihood functions, adds the data and model the
    likelihood is evaluated against.

    Notes
    -----
    Subclasses implement `data` and `model` in addition to the
    `DensityInterface` contract.
    """

    @property
    @abstractmethod
    def data(self) -> Data:
        ...

    @property
    @abstractmethod
    def model(self) -> ForwardMap:
        ...


class CovarianceOperatorInterface(ABC):
    """
    Interface for objects that apply a covariance operator and its Cholesky
    factors to a vector, without necessarily exposing the operator's full
    structure (dense, diagonal, or otherwise).

    Notes
    -----
    Subclasses implement `apply_chol_factor`, `apply_chol_factor_transpose`,
    and `apply_inverse`.
    """

    @abstractmethod
    def apply_chol_factor(self, x: ndarray) -> ndarray:
        ...

    @abstractmethod
    def apply_chol_factor_transpose(self, x: ndarray) -> ndarray:
        ...

    @abstractmethod
    def apply_inverse(self, x: ndarray) -> ndarray:
        ...


class CovarianceFunctionInterface(ABC):
    """
    Interface for stationary or non-stationary covariance functions
    evaluated pairwise between two sets of points.

    Notes
    -----
    Subclasses implement `evaluate_covariance`. `MaternCovariance2D` is
    curated but does not formally implement this interface, despite
    matching its contract structurally. `isinstance` checks against this
    interface will fail for `MaternCovariance2D` instances.
    """

    @abstractmethod
    def evaluate_covariance(self, xGrid: ndarray, yGrid: ndarray) -> ndarray:
        ...


class BayesianModelInterface(ABC):
    """
    Interface for a Bayesian model exposing its likelihood and prior as
    separate components.

    Notes
    -----
    Subclasses implement `likelihood` and `prior`.
    """

    @property
    @abstractmethod
    def likelihood(self) -> DensityInterface:
        ...

    @property
    @abstractmethod
    def prior(self) -> ProbabilityMeasure:
        ...


class Predictor(ABC):
    """
    Interface for out-of-sample forward predictions.
    """

    @abstractmethod
    def mean(self) -> ndarray:
        """
        Estimate the mean of the predictor at the query sites.
        """
        ...
