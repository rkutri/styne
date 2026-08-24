from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


class BackendCapabilityError(NotImplementedError):
    """Raised when a backend does not provide an optional capability."""


@dataclass(frozen=True)
class BackendMetadata:
    """Array metadata that must survive backend-neutral operations."""

    dtype: Any
    device: Any = None


@dataclass(frozen=True)
class BackendCapabilities:
    """Capabilities whose availability differs between backends."""

    automaticDifferentiation: bool = False
    compilation: bool = False
    vectorisation: bool = False
    controlFlow: bool = False
    spectralTransforms: bool = False
    transformedLoops: bool = False


class ArrayNamespace:
    """Small array operation set owned by `styne`."""

    def abs(self, array):
        raise NotImplementedError

    def all(self, array, *, axis=None):
        raise NotImplementedError

    def any(self, array, *, axis=None):
        raise NotImplementedError

    def broadcast_to(self, array, shape):
        raise NotImplementedError

    def clip(self, array, minimum, maximum):
        raise NotImplementedError

    def concatenate(self, arrays, *, axis=0):
        raise NotImplementedError

    def diagonal(self, array, *, axis1=-2, axis2=-1):
        raise NotImplementedError

    def exp(self, array):
        raise NotImplementedError

    def expand_dims(self, array, *, axis):
        raise NotImplementedError

    def isfinite(self, array):
        raise NotImplementedError

    def log(self, array):
        raise NotImplementedError

    def logaddexp(self, first, second):
        raise NotImplementedError

    def maximum(self, first, second):
        raise NotImplementedError

    def minimum(self, first, second):
        raise NotImplementedError

    def mean(self, array, *, axis=None):
        raise NotImplementedError

    def norm(self, array, *, axis=None):
        raise NotImplementedError

    def prod(self, array, *, axis=None):
        raise NotImplementedError

    def sigmoid(self, array):
        raise NotImplementedError

    def sqrt(self, array):
        raise NotImplementedError

    def square(self, array):
        raise NotImplementedError

    def stack(self, arrays, *, axis=0):
        raise NotImplementedError

    def sum(self, array, *, axis=None):
        raise NotImplementedError

    def where(self, condition, first, second):
        raise NotImplementedError

    def gammaln(self, array):
        raise NotImplementedError

    def gamma(self, array):
        raise NotImplementedError

    def bessel_kv(self, order, array):
        raise NotImplementedError

    def logsumexp(self, array, *, axis=None):
        raise NotImplementedError


class Backend(ABC):
    """Backend contract for graph-native numerical operations."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def namespace(self) -> ArrayNamespace:
        ...

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    @abstractmethod
    def is_array(self, value: Any) -> bool:
        ...

    @abstractmethod
    def metadata(self, array) -> BackendMetadata:
        ...

    def asarray(self, value, *, dtype=None, device=None):
        raise NotImplementedError

    def zeros(self, shape, *, dtype=None, device=None):
        raise NotImplementedError

    def ones(self, shape, *, dtype=None, device=None):
        raise NotImplementedError

    def full(self, shape, fillValue, *, dtype=None, device=None):
        raise NotImplementedError

    def eye(self, dimension, *, dtype=None, device=None):
        raise NotImplementedError

    def arange(self, start, stop=None, step=1, *, dtype=None, device=None):
        raise NotImplementedError

    def linspace(self, start, stop, number, *, dtype=None, device=None):
        raise NotImplementedError

    def random_state(self, seed=None, *, device=None):
        """Create backend random state from a seed."""
        raise NotImplementedError

    def normal(self, randomState, shape, *, dtype=None, device=None):
        """Return a normal sample and the updated random state."""
        raise NotImplementedError

    def uniform(self, randomState, shape, *, dtype=None, device=None):
        """Return a uniform sample and the updated random state."""
        raise NotImplementedError

    def poisson(self, randomState, rate, shape=None):
        """Return a Poisson sample and the updated random state."""
        raise NotImplementedError

    def binomial(self, randomState, trials, probability, shape=None):
        """Return a binomial sample and the updated random state."""
        raise NotImplementedError

    def grad(self, function: Callable, *, argnums=0):
        self._unsupported("automatic differentiation")

    def value_and_grad(self, function: Callable, *, argnums=0):
        self._unsupported("automatic differentiation")

    def jacobian(self, function: Callable, *, argnums=0):
        self._unsupported("automatic differentiation")

    def hessian(self, function: Callable, *, argnums=0):
        self._unsupported("automatic differentiation")

    def jvp(self, function: Callable, primals, tangents):
        self._unsupported("automatic differentiation")

    def vjp(self, function: Callable, *primals):
        self._unsupported("automatic differentiation")

    def compile(self, function: Callable, **options):
        self._unsupported("compilation")

    def vectorize(self, function: Callable, *, inAxes=0, outAxes=0):
        self._unsupported("vectorisation")

    def cond(
            self, predicate, trueFunction: Callable,
            falseFunction: Callable, operand):
        self._unsupported("control flow")

    def cholesky(self, matrix):
        raise NotImplementedError

    def solve(self, matrix, rightHandSide):
        raise NotImplementedError

    def solve_triangular(self, matrix, rightHandSide, *, lower):
        raise NotImplementedError

    def dct1(self, array, *, axis=-1):
        self._unsupported("spectral transforms")

    def dst1(self, array, *, axis=-1):
        self._unsupported("spectral transforms")

    def scan(self, function: Callable, initialValue, inputs, *, length=None):
        self._unsupported("transformed loops")

    def _unsupported(self, capability: str):
        raise BackendCapabilityError(
            f"Backend {self.name!r} does not support {capability}."
        )
