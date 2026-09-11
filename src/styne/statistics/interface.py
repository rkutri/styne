from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Sequence, TypeAlias

from numpy import ndarray
from numpy.random import Generator, SeedSequence

from styne.model.forwardmap import ForwardMap
from styne.parameter.parameter import Parameter
from styne.statistics.measure import ProbabilityMeasure
from styne.statistics.data import Data


SeedType = Union[int, Sequence[int], SeedSequence, None]
BackendArray: TypeAlias = Any


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
    return a rank-zero array owned by the state backend. It may be a properly
    normalised log-density or one missing an additive
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
    def evaluate_log(self, state: Parameter) -> BackendArray:
        ...


class RadonNikodymInterface(DensityInterface):
    """Density represented by an RN factor and its reference measure.

    Samplers such as pCN consume the derivative for acceptance while using
    the reference to construct proposals. ``evaluate_log`` remains the full
    target density supplied by the concrete implementation.
    """

    @property
    @abstractmethod
    def reference(self) -> ProbabilityMeasure:
        ...

    @property
    @abstractmethod
    def derivative(self) -> DensityInterface:
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
    Interface for backend-native covariance operator applications.

    A covariance operator may use dense, diagonal, or other internal
    structure. Its numerical inputs and outputs are backend arrays: NumPy,
    PyTorch, or JAX arrays remain owned by their originating backend. The
    operator dimension is structural Python metadata and is therefore an
    ``int``. Implementations that additionally expose scalar numerical
    quantities, such as log determinants or quadratic forms, return rank-zero
    backend arrays rather than Python ``float`` values.

    Notes
    -----
    Subclasses implement `apply_chol_factor`, `apply_chol_factor_transpose`,
    and `apply_inverse`. Each operation accepts a vector or batch of vectors
    in the backend's usual trailing-coordinate layout and returns an array of
    the corresponding shape.
    """

    @abstractmethod
    def apply_chol_factor(self, x: BackendArray) -> BackendArray:
        ...

    @abstractmethod
    def apply_chol_factor_transpose(self, x: BackendArray) -> BackendArray:
        ...

    @abstractmethod
    def apply_inverse(self, x: BackendArray) -> BackendArray:
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
