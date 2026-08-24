import jax
import jax.numpy as jnp
import jax.scipy as jsp

from styne.backend.interface import (
    ArrayNamespace,
    Backend,
    BackendCapabilities,
    BackendCapabilityError,
    BackendMetadata,
)


def _register_parameter_containers():
    """Register Styne's public parameters as JAX pytrees."""
    from styne.parameter.block import BlockParameter
    from styne.parameter.function import Function
    from styne.parameter.scalar import Scalar
    from styne.parameter.vector import Vector

    jax.tree_util.register_pytree_node(
        Vector,
        lambda parameter: ((parameter.coordinate,), None),
        lambda metadata, children: Vector(children[0]),
    )
    jax.tree_util.register_pytree_node(
        Scalar,
        lambda parameter: ((parameter.coordinate,), None),
        lambda metadata, children: Scalar(children[0]),
    )
    jax.tree_util.register_pytree_node(
        Function,
        lambda parameter: (
            (parameter.coordinate,), parameter.expansion
        ),
        lambda expansion, children: Function(children[0], expansion),
    )
    jax.tree_util.register_pytree_node(
        BlockParameter,
        lambda parameter: (
            tuple(
                parameter.block(index)
                for index in range(parameter.nBlocks)
            ),
            tuple(parameter.names.items()),
        ),
        lambda names, children: BlockParameter(
            list(children), dict(names)
        ),
    )


class JAXNamespace(ArrayNamespace):
    """Declared `styne` array operations implemented by JAX."""

    def abs(self, array):
        return jnp.abs(array)

    def all(self, array, *, axis=None):
        return jnp.all(array, axis=axis)

    def any(self, array, *, axis=None):
        return jnp.any(array, axis=axis)

    def broadcast_to(self, array, shape):
        return jnp.broadcast_to(array, shape)

    def clip(self, array, minimum, maximum):
        return jnp.clip(array, minimum, maximum)

    def concatenate(self, arrays, *, axis=0):
        return jnp.concatenate(arrays, axis=axis)

    def diagonal(self, array, *, axis1=-2, axis2=-1):
        return jnp.diagonal(array, axis1=axis1, axis2=axis2)

    def exp(self, array):
        return jnp.exp(array)

    def expand_dims(self, array, *, axis):
        return jnp.expand_dims(array, axis=axis)

    def isfinite(self, array):
        return jnp.isfinite(array)

    def log(self, array):
        return jnp.log(array)

    def logaddexp(self, first, second):
        return jnp.logaddexp(first, second)

    def maximum(self, first, second):
        return jnp.maximum(first, second)

    def minimum(self, first, second):
        return jnp.minimum(first, second)

    def mean(self, array, *, axis=None):
        return jnp.mean(array, axis=axis)

    def norm(self, array, *, axis=None):
        return jnp.linalg.norm(array, axis=axis)

    def prod(self, array, *, axis=None):
        return jnp.prod(array, axis=axis)

    def sigmoid(self, array):
        return jax.nn.sigmoid(array)

    def sqrt(self, array):
        return jnp.sqrt(array)

    def square(self, array):
        return jnp.square(array)

    def stack(self, arrays, *, axis=0):
        return jnp.stack(arrays, axis=axis)

    def swapaxes(self, array, axis1, axis2):
        return jnp.swapaxes(array, axis1, axis2)

    def sum(self, array, *, axis=None):
        return jnp.sum(array, axis=axis)

    def where(self, condition, first, second):
        return jnp.where(condition, first, second)

    def gammaln(self, array):
        return jsp.special.gammaln(array)

    def gamma(self, array):
        return jsp.special.gamma(array)

    def bessel_kv(self, order, array):
        raise BackendCapabilityError(
            "JAX does not provide arbitrary-order modified Bessel K."
        )

    def logsumexp(self, array, *, axis=None):
        return jsp.special.logsumexp(array, axis=axis)


class JAXBackend(Backend):
    """Optional graph-native backend implemented with JAX."""

    _namespace = JAXNamespace()
    _capabilities = BackendCapabilities(
        automaticDifferentiation=True,
        compilation=True,
        vectorisation=True,
        controlFlow=True,
        spectralTransforms=True,
        transformedLoops=True,
    )
    _parameterContainersRegistered = False

    def __init__(self):
        if not type(self)._parameterContainersRegistered:
            _register_parameter_containers()
            type(self)._parameterContainersRegistered = True

    @property
    def name(self):
        return "jax"

    @property
    def namespace(self):
        return self._namespace

    @property
    def capabilities(self):
        return self._capabilities

    def is_array(self, value):
        return isinstance(value, (jax.Array, jax.core.Tracer))

    def metadata(self, array):
        if not self.is_array(array):
            raise TypeError("JAX metadata requires a JAX array or tracer.")
        device = getattr(array, "device", None)
        if callable(device):
            device = device()
        return BackendMetadata(dtype=array.dtype, device=device)

    def asarray(self, value, *, dtype=None, device=None):
        return self._place(jnp.asarray(value, dtype=dtype), device)

    def zeros(self, shape, *, dtype=None, device=None):
        return self._place(jnp.zeros(shape, dtype=dtype), device)

    def ones(self, shape, *, dtype=None, device=None):
        return self._place(jnp.ones(shape, dtype=dtype), device)

    def full(self, shape, fillValue, *, dtype=None, device=None):
        return self._place(jnp.full(shape, fillValue, dtype=dtype), device)

    def eye(self, dimension, *, dtype=None, device=None):
        return self._place(jnp.eye(dimension, dtype=dtype), device)

    def arange(self, start, stop=None, step=1, *, dtype=None, device=None):
        if stop is None:
            array = jnp.arange(start, dtype=dtype)
        else:
            array = jnp.arange(start, stop, step, dtype=dtype)
        return self._place(array, device)

    def linspace(self, start, stop, number, *, dtype=None, device=None):
        array = jnp.linspace(start, stop, number, dtype=dtype)
        return self._place(array, device)

    def random_state(self, seed=None, *, device=None):
        if seed is None:
            raise ValueError("JAX random state requires an explicit seed.")
        return self._place(jax.random.key(seed), device)

    def normal(self, randomState, shape, *, dtype=None, device=None):
        sampleKey, nextState = jax.random.split(randomState)
        sample = jax.random.normal(
            sampleKey, self._normalise_shape(shape), dtype=dtype
        )
        return self._place(sample, device), self._place(nextState, device)

    def uniform(self, randomState, shape, *, dtype=None, device=None):
        sampleKey, nextState = jax.random.split(randomState)
        sample = jax.random.uniform(
            sampleKey, self._normalise_shape(shape), dtype=dtype
        )
        return self._place(sample, device), self._place(nextState, device)

    def poisson(self, randomState, rate, shape=None):
        sampleKey, nextState = jax.random.split(randomState)
        sample = jax.random.poisson(sampleKey, rate, shape=shape)
        return sample, nextState

    def binomial(self, randomState, trials, probability, shape=None):
        sampleKey, nextState = jax.random.split(randomState)
        sample = jax.random.binomial(
            sampleKey, n=trials, p=probability, shape=shape
        )
        return sample, nextState

    def grad(self, function, *, argnums=0):
        return jax.grad(function, argnums=argnums)

    def value_and_grad(self, function, *, argnums=0):
        return jax.value_and_grad(function, argnums=argnums)

    def jacobian(self, function, *, argnums=0):
        return jax.jacobian(function, argnums=argnums)

    def hessian(self, function, *, argnums=0):
        return jax.hessian(function, argnums=argnums)

    def jvp(self, function, primals, tangents):
        return jax.jvp(function, primals, tangents)

    def vjp(self, function, *primals):
        return jax.vjp(function, *primals)

    def compile(self, function, **options):
        return jax.jit(function, **options)

    def vectorize(self, function, *, inAxes=0, outAxes=0):
        return jax.vmap(function, in_axes=inAxes, out_axes=outAxes)

    def cond(
            self, predicate, trueFunction,
            falseFunction, operand):
        return jax.lax.cond(
            predicate, trueFunction, falseFunction, operand
        )

    def cholesky(self, matrix):
        return jnp.linalg.cholesky(matrix)

    def solve(self, matrix, rightHandSide):
        return jnp.linalg.solve(matrix, rightHandSide)

    def solve_triangular(self, matrix, rightHandSide, *, lower):
        return jsp.linalg.solve_triangular(
            matrix, rightHandSide, lower=lower
        )

    def dct1(self, array, *, axis=-1):
        if array.shape[axis] < 2:
            raise ValueError("DCT-I requires an input length of at least two.")
        # JAX exposes no DCT-I.  The even extension has length 2*(N-1),
        # which is the defining type-I grid; using a native DCT-II here would
        # shift the DNA modes and corrupt endpoint values.  Since the
        # extension is real, rFFT retains exactly the required N bins.
        moved = jnp.moveaxis(array, axis, -1)
        extension = jnp.concatenate((moved, moved[..., -2:0:-1]), axis=-1)
        transformed = jnp.fft.rfft(extension, axis=-1)
        result = jnp.real(transformed)
        return jnp.moveaxis(result, -1, axis)

    def dst1(self, array, *, axis=-1):
        # The odd extension has length 2*(N+1), reproducing unnormalised
        # DST-I and its zero endpoints exactly; only its one-sided spectrum
        # is required.
        moved = jnp.moveaxis(array, axis, -1)
        zero = jnp.zeros_like(moved[..., :1])
        extension = jnp.concatenate(
            (zero, moved, zero, -moved[..., ::-1]), axis=-1
        )
        transformed = -jnp.imag(jnp.fft.rfft(extension, axis=-1))
        result = transformed[..., 1:moved.shape[-1] + 1]
        return jnp.moveaxis(result, -1, axis)

    def scan(self, function, initialValue, inputs, *, length=None):
        return jax.lax.scan(function, initialValue, inputs, length=length)

    @staticmethod
    def _normalise_shape(shape):
        return (shape,) if isinstance(shape, int) else shape

    @staticmethod
    def _resolve_device(device):
        if device is None or not isinstance(device, str):
            return device
        devices = jax.devices(device)
        if not devices:
            raise ValueError(f"No JAX device is available for {device!r}.")
        return devices[0]

    @classmethod
    def _place(cls, array, device):
        resolvedDevice = cls._resolve_device(device)
        if resolvedDevice is None:
            return array
        return jax.device_put(array, resolvedDevice)
