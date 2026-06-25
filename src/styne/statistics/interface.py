from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, runtime_checkable, Protocol

from numpy import ndarray
from numpy.random import Generator, SeedSequence

from styne.model.model import Model
from styne.parameter.parameter import Parameter
from styne.statistics.measure import ProbabilityMeasure
from styne.statistics.data import Data


SeedType = Union[int, Sequence[int], SeedSequence, None]


class DensityInterface(ABC):

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

    @property
    @abstractmethod
    def data(self) -> Data:
        ...

    @property
    @abstractmethod
    def model(self) -> Model:
        ...


class CovarianceOperatorInterface(ABC):

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

    @abstractmethod
    def evaluate_covariance(self, xGrid: ndarray, yGrid: ndarray) -> ndarray:
        ...


class BayesianModelInterface(ABC):

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
