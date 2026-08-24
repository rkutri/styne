import torch
from torch.utils import _pytree

from styne.backend.interface import (
    ArrayNamespace,
    Backend,
    BackendCapabilities,
    BackendCapabilityError,
    BackendMetadata,
)


class PyTorchNamespace(ArrayNamespace):
    """Declared `styne` array operations implemented by PyTorch."""

    def abs(self, array):
        return torch.abs(array)

    def all(self, array, *, axis=None):
        if axis is None:
            return torch.all(array)
        return torch.all(array, dim=axis)

    def any(self, array, *, axis=None):
        if axis is None:
            return torch.any(array)
        return torch.any(array, dim=axis)

    def broadcast_to(self, array, shape):
        return torch.broadcast_to(array, shape)

    def clip(self, array, minimum, maximum):
        return torch.clamp(array, min=minimum, max=maximum)

    def concatenate(self, arrays, *, axis=0):
        return torch.cat(tuple(arrays), dim=axis)

    def diagonal(self, array, *, axis1=-2, axis2=-1):
        return torch.diagonal(array, dim1=axis1, dim2=axis2)

    def exp(self, array):
        return torch.exp(array)

    def expand_dims(self, array, *, axis):
        return torch.unsqueeze(array, dim=axis)

    def isfinite(self, array):
        return torch.isfinite(array)

    def log(self, array):
        return torch.log(array)

    def logaddexp(self, first, second):
        reference = first if isinstance(first, torch.Tensor) else second
        return torch.logaddexp(
            self._as_like(first, reference),
            self._as_like(second, reference),
        )

    def maximum(self, first, second):
        return torch.maximum(first, self._as_like(second, first))

    def minimum(self, first, second):
        return torch.minimum(first, self._as_like(second, first))

    def mean(self, array, *, axis=None):
        if axis is None:
            return torch.mean(array)
        return torch.mean(array, dim=axis)

    def norm(self, array, *, axis=None):
        return torch.linalg.vector_norm(array, dim=axis)

    def prod(self, array, *, axis=None):
        if axis is None:
            return torch.prod(array)
        return torch.prod(array, dim=axis)

    def sigmoid(self, array):
        return torch.sigmoid(array)

    def sqrt(self, array):
        return torch.sqrt(array)

    def square(self, array):
        return torch.square(array)

    def stack(self, arrays, *, axis=0):
        return torch.stack(tuple(arrays), dim=axis)

    def swapaxes(self, array, axis1, axis2):
        return torch.swapaxes(array, axis1, axis2)

    def sum(self, array, *, axis=None):
        if axis is None:
            return torch.sum(array)
        return torch.sum(array, dim=axis)

    def where(self, condition, first, second):
        reference = first if isinstance(first, torch.Tensor) else second
        return torch.where(
            condition,
            self._as_like(first, reference),
            self._as_like(second, reference),
        )

    def gammaln(self, array):
        return torch.special.gammaln(array)

    def gamma(self, array):
        return torch.exp(torch.lgamma(array))

    def bessel_kv(self, order, array):
        raise BackendCapabilityError(
            "PyTorch does not provide arbitrary-order modified Bessel K."
        )

    def logsumexp(self, array, *, axis=None):
        if axis is None:
            return torch.logsumexp(array.reshape(-1), dim=0)
        return torch.logsumexp(array, dim=axis)

    @staticmethod
    def _as_like(value, reference):
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(
            value, dtype=reference.dtype, device=reference.device
        )


class PyTorchBackend(Backend):
    """Optional graph-native backend implemented with PyTorch."""

    _namespace = PyTorchNamespace()
    _capabilities = BackendCapabilities(
        automaticDifferentiation=True,
        compilation=True,
        vectorisation=True,
        controlFlow=True,
        spectralTransforms=True,
        # Dynamo cannot represent explicit torch.Generator state in one graph.
        transformedLoops=False,
    )
    _parameterContainersRegistered = False

    def __init__(self):
        if not type(self)._parameterContainersRegistered:
            self._register_parameter_containers()
            type(self)._parameterContainersRegistered = True

    @staticmethod
    def _register_parameter_containers():
        from styne.mcmc.transition import EvaluatedState, TransitionData
        from styne.parameter.block import BlockParameter
        from styne.parameter.function import Function
        from styne.parameter.scalar import Scalar
        from styne.parameter.vector import Vector

        _pytree.register_pytree_node(
            Vector,
            lambda parameter: ((parameter.coordinate,), None),
            lambda children, metadata: Vector(children[0]),
        )
        _pytree.register_pytree_node(
            Scalar,
            lambda parameter: ((parameter.coordinate,), None),
            lambda children, metadata: Scalar(children[0]),
        )
        _pytree.register_pytree_node(
            Function,
            lambda parameter: (
                (parameter.coordinate,), parameter.expansion
            ),
            lambda children, expansion: Function(
                children[0], expansion
            ),
        )
        _pytree.register_pytree_node(
            BlockParameter,
            lambda parameter: (
                tuple(
                    parameter.block(index)
                    for index in range(parameter.nBlocks)
                ),
                tuple(parameter.names.items()),
            ),
            lambda children, names: BlockParameter(
                list(children), dict(names)
            ),
        )
        _pytree.register_pytree_node(
            EvaluatedState,
            lambda state: ((state.parameter, state.logDensity), None),
            lambda children, metadata: EvaluatedState(*children),
        )
        _pytree.register_pytree_node(
            TransitionData,
            lambda transition: (
                (
                    transition.current,
                    transition.proposed,
                    transition.outcome,
                    transition.logAcceptanceProbability,
                    transition.auxiliary,
                ),
                None,
            ),
            lambda children, metadata: TransitionData(
                current=children[0],
                proposed=children[1],
                outcome=children[2],
                logAcceptanceProbability=children[3],
                auxiliary=children[4],
            ),
        )

    @property
    def name(self):
        return "pytorch"

    @property
    def namespace(self):
        return self._namespace

    @property
    def capabilities(self):
        return self._capabilities

    def is_array(self, value):
        return isinstance(value, torch.Tensor)

    def metadata(self, array):
        if not self.is_array(array):
            raise TypeError("PyTorch metadata requires a torch.Tensor.")
        return BackendMetadata(dtype=array.dtype, device=array.device)

    def asarray(self, value, *, dtype=None, device=None):
        return torch.as_tensor(
            value, dtype=self._resolve_dtype(dtype), device=device
        )

    def zeros(self, shape, *, dtype=None, device=None):
        return torch.zeros(
            shape, dtype=self._resolve_dtype(dtype), device=device
        )

    def ones(self, shape, *, dtype=None, device=None):
        return torch.ones(
            shape, dtype=self._resolve_dtype(dtype), device=device
        )

    def full(self, shape, fillValue, *, dtype=None, device=None):
        return torch.full(
            self._normalise_shape(shape),
            fillValue,
            dtype=self._resolve_dtype(dtype),
            device=device,
        )

    def eye(self, dimension, *, dtype=None, device=None):
        return torch.eye(
            dimension, dtype=self._resolve_dtype(dtype), device=device
        )

    def arange(self, start, stop=None, step=1, *, dtype=None, device=None):
        resolvedDtype = self._resolve_dtype(dtype)
        if stop is None:
            return torch.arange(start, dtype=resolvedDtype, device=device)
        return torch.arange(
            start, stop, step, dtype=resolvedDtype, device=device
        )

    def linspace(self, start, stop, number, *, dtype=None, device=None):
        return torch.linspace(
            start,
            stop,
            number,
            dtype=self._resolve_dtype(dtype),
            device=device,
        )

    def random_state(self, seed=None, *, device=None):
        generator = torch.Generator(device=device or "cpu")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        return generator

    def normal(self, randomState, shape, *, dtype=None, device=None):
        sample = torch.randn(
            self._normalise_shape(shape),
            dtype=self._resolve_dtype(dtype),
            device=device or randomState.device,
            generator=randomState,
        )
        return sample, randomState

    def uniform(self, randomState, shape, *, dtype=None, device=None):
        sample = torch.rand(
            self._normalise_shape(shape),
            dtype=self._resolve_dtype(dtype),
            device=device or randomState.device,
            generator=randomState,
        )
        return sample, randomState

    def poisson(self, randomState, rate, shape=None):
        rateTensor = self._random_tensor(rate, randomState)
        if shape is not None:
            rateTensor = torch.broadcast_to(
                rateTensor, self._normalise_shape(shape)
            )
        return torch.poisson(rateTensor, generator=randomState), randomState

    def binomial(self, randomState, trials, probability, shape=None):
        probabilityTensor = self._random_tensor(probability, randomState)
        trialsTensor = self._random_tensor(
            trials, randomState, dtype=probabilityTensor.dtype
        )
        if shape is not None:
            shape = self._normalise_shape(shape)
            probabilityTensor = torch.broadcast_to(probabilityTensor, shape)
            trialsTensor = torch.broadcast_to(trialsTensor, shape)
        sample = torch.binomial(
            trialsTensor, probabilityTensor, generator=randomState
        )
        return sample, randomState

    def grad(self, function, *, argnums=0):
        return torch.func.grad(function, argnums=argnums)

    def value_and_grad(self, function, *, argnums=0):
        gradAndValue = torch.func.grad_and_value(function, argnums=argnums)

        def value_and_grad_function(*args, **kwargs):
            gradient, value = gradAndValue(*args, **kwargs)
            return value, gradient

        return value_and_grad_function

    def jacobian(self, function, *, argnums=0):
        return torch.func.jacrev(function, argnums=argnums)

    def hessian(self, function, *, argnums=0):
        return torch.func.hessian(function, argnums=argnums)

    def jvp(self, function, primals, tangents):
        return torch.func.jvp(function, primals, tangents)

    def vjp(self, function, *primals):
        return torch.func.vjp(function, *primals)

    def compile(self, function, **options):
        return torch.compile(function, **options)

    def vectorize(self, function, *, inAxes=0, outAxes=0):
        return torch.vmap(function, in_dims=inAxes, out_dims=outAxes)

    def cond(
            self, predicate, trueFunction,
            falseFunction, operand):
        return torch.cond(
            predicate, trueFunction, falseFunction, (operand,)
        )

    def scan(self, function, initialValue, inputs, *, length=None):
        if inputs is not None:
            raise NotImplementedError(
                "PyTorch scan requires inputs=None and a static length."
            )
        if length is None or length < 1:
            raise ValueError("PyTorch scan requires a positive static length.")

        value = initialValue
        outputs = []
        for _ in range(length):
            value, output = function(value, None)
            outputs.append(output)

        flattened, structure = _pytree.tree_flatten(outputs[0])
        flattenedOutputs = [
            _pytree.tree_flatten(output)[0] for output in outputs
        ]
        stacked = [
            torch.stack([output[index] for output in flattenedOutputs])
            for index in range(len(flattened))
        ]
        return value, _pytree.tree_unflatten(stacked, structure)

    def cholesky(self, matrix):
        return torch.linalg.cholesky(matrix)

    def solve(self, matrix, rightHandSide):
        return torch.linalg.solve(matrix, rightHandSide)

    def solve_triangular(self, matrix, rightHandSide, *, lower):
        vectorRightHandSide = rightHandSide.ndim == matrix.ndim - 1
        if vectorRightHandSide:
            rightHandSide = rightHandSide.unsqueeze(-1)
        solution = torch.linalg.solve_triangular(
            matrix, rightHandSide, upper=not lower
        )
        if vectorRightHandSide:
            return solution.squeeze(-1)
        return solution

    def dct1(self, array, *, axis=-1):
        if array.shape[axis] < 2:
            raise ValueError("DCT-I requires an input length of at least two.")
        # PyTorch exposes no DCT-I.  This even extension has length 2*(N-1);
        # torch-native DCT-II conventions would shift the DNA modes.  rFFT
        # retains the complete nonredundant spectrum of this real extension.
        moved = torch.movedim(array, axis, -1)
        reflectedInterior = torch.flip(moved[..., 1:-1], dims=(-1,))
        extension = torch.cat((moved, reflectedInterior), dim=-1)
        transformed = torch.fft.rfft(extension, dim=-1)
        result = torch.real(transformed)
        return torch.movedim(result, -1, axis)

    def dst1(self, array, *, axis=-1):
        # The 2*(N+1) odd extension gives unnormalised DST-I with exact zero
        # endpoints, matching the retained 0.2.1 synthesis convention; only
        # the one-sided spectrum is required.
        moved = torch.movedim(array, axis, -1)
        zero = torch.zeros_like(moved[..., :1])
        extension = torch.cat(
            (zero, moved, zero, -torch.flip(moved, dims=(-1,))), dim=-1
        )
        transformed = -torch.imag(torch.fft.rfft(extension, dim=-1))
        result = transformed[..., 1:moved.shape[-1] + 1]
        return torch.movedim(result, -1, axis)

    @staticmethod
    def _normalise_shape(shape):
        return (shape,) if isinstance(shape, int) else shape

    @staticmethod
    def _random_tensor(value, randomState, dtype=None):
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(
            value, dtype=dtype, device=randomState.device
        )

    @staticmethod
    def _resolve_dtype(dtype):
        if not isinstance(dtype, str):
            return dtype
        resolvedDtype = getattr(torch, dtype, None)
        if not isinstance(resolvedDtype, torch.dtype):
            raise TypeError(f"Unknown PyTorch dtype {dtype!r}.")
        return resolvedDtype
