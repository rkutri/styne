import numpy as np
from scipy import fft, linalg, special

from styne.backend.interface import (
    ArrayNamespace,
    Backend,
    BackendCapabilities,
    BackendMetadata,
)


class NumPyNamespace(ArrayNamespace):
    """Declared `styne` array operations implemented by NumPy and SciPy."""

    def abs(self, array):
        return np.abs(array)

    def all(self, array, *, axis=None):
        return np.all(array, axis=axis)

    def any(self, array, *, axis=None):
        return np.any(array, axis=axis)

    def broadcast_to(self, array, shape):
        return np.broadcast_to(array, shape)

    def clip(self, array, minimum, maximum):
        return np.clip(array, minimum, maximum)

    def concatenate(self, arrays, *, axis=0):
        return np.concatenate(arrays, axis=axis)

    def diagonal(self, array, *, axis1=-2, axis2=-1):
        return np.diagonal(array, axis1=axis1, axis2=axis2)

    def exp(self, array):
        return np.exp(array)

    def expand_dims(self, array, *, axis):
        return np.expand_dims(array, axis=axis)

    def isfinite(self, array):
        return np.isfinite(array)

    def log(self, array):
        return np.log(array)

    def logaddexp(self, first, second):
        return np.logaddexp(first, second)

    def maximum(self, first, second):
        return np.maximum(first, second)

    def minimum(self, first, second):
        return np.minimum(first, second)

    def mean(self, array, *, axis=None):
        return np.mean(array, axis=axis)

    def norm(self, array, *, axis=None):
        return np.linalg.norm(array, axis=axis)

    def prod(self, array, *, axis=None):
        return np.prod(array, axis=axis)

    def sigmoid(self, array):
        return special.expit(array)

    def sqrt(self, array):
        return np.sqrt(array)

    def square(self, array):
        return np.square(array)

    def stack(self, arrays, *, axis=0):
        return np.stack(arrays, axis=axis)

    def sum(self, array, *, axis=None):
        return np.sum(array, axis=axis)

    def where(self, condition, first, second):
        return np.where(condition, first, second)

    def gammaln(self, array):
        return special.gammaln(array)

    def gamma(self, array):
        return special.gamma(array)

    def bessel_kv(self, order, array):
        return special.kv(order, array)

    def logsumexp(self, array, *, axis=None):
        return special.logsumexp(array, axis=axis)


class NumPyBackend(Backend):
    """Reference backend implemented with NumPy and SciPy."""

    _namespace = NumPyNamespace()
    _capabilities = BackendCapabilities(
        controlFlow=True,
        spectralTransforms=True,
    )

    @property
    def name(self):
        return "numpy"

    @property
    def namespace(self):
        return self._namespace

    @property
    def capabilities(self):
        return self._capabilities

    def is_array(self, value):
        return isinstance(value, (np.ndarray, np.generic))

    def metadata(self, array):
        if not self.is_array(array):
            raise TypeError(
                "NumPy metadata requires a numpy.ndarray or numpy scalar."
            )
        return BackendMetadata(dtype=array.dtype, device="cpu")

    def asarray(self, value, *, dtype=None, device=None):
        self._require_cpu(device)
        return np.asarray(value, dtype=dtype)

    def zeros(self, shape, *, dtype=None, device=None):
        self._require_cpu(device)
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, *, dtype=None, device=None):
        self._require_cpu(device)
        return np.ones(shape, dtype=dtype)

    def full(self, shape, fillValue, *, dtype=None, device=None):
        self._require_cpu(device)
        return np.full(shape, fillValue, dtype=dtype)

    def eye(self, dimension, *, dtype=None, device=None):
        self._require_cpu(device)
        return np.eye(dimension, dtype=dtype)

    def arange(self, start, stop=None, step=1, *, dtype=None, device=None):
        self._require_cpu(device)
        if stop is None:
            return np.arange(start, dtype=dtype)
        return np.arange(start, stop, step, dtype=dtype)

    def linspace(self, start, stop, number, *, dtype=None, device=None):
        self._require_cpu(device)
        return np.linspace(start, stop, number, dtype=dtype)

    def random_state(self, seed=None, *, device=None):
        self._require_cpu(device)
        return np.random.default_rng(seed)

    def normal(self, randomState, shape, *, dtype=None, device=None):
        self._require_cpu(device)
        sample = randomState.standard_normal(size=shape)
        return np.asarray(sample, dtype=dtype), randomState

    def uniform(self, randomState, shape, *, dtype=None, device=None):
        self._require_cpu(device)
        sample = randomState.uniform(size=shape)
        return np.asarray(sample, dtype=dtype), randomState

    def poisson(self, randomState, rate, shape=None):
        return randomState.poisson(rate, size=shape), randomState

    def binomial(self, randomState, trials, probability, shape=None):
        return (
            randomState.binomial(trials, probability, size=shape),
            randomState,
        )

    def cond(
            self, predicate, trueFunction,
            falseFunction, operand):
        function = trueFunction if bool(predicate) else falseFunction
        return function(operand)

    def cholesky(self, matrix):
        return np.linalg.cholesky(matrix)

    def solve(self, matrix, rightHandSide):
        return np.linalg.solve(matrix, rightHandSide)

    def solve_triangular(self, matrix, rightHandSide, *, lower):
        return linalg.solve_triangular(
            matrix, rightHandSide, lower=lower
        )

    def dct1(self, array, *, axis=-1):
        return fft.dct(array, type=1, norm="backward", axis=axis)

    def dst1(self, array, *, axis=-1):
        return fft.dst(array, type=1, norm="backward", axis=axis)

    @staticmethod
    def _require_cpu(device):
        if device not in (None, "cpu"):
            raise ValueError(
                f"NumPy supports only the CPU device, received {device!r}."
            )
