import pytest


torch = pytest.importorskip("torch", reason="PyTorch is an optional backend")

from styne.backend import (  # noqa: E402
    BackendCapabilityError,
    get_backend,
    infer_backend,
)
from styne.backend.pytorch import (  # noqa: E402
    PyTorchBackend,
    PyTorchNamespace,
)


def test_pytorch_backend_is_registered_for_tensors():
    backend = get_backend("pytorch")
    tensor = torch.zeros(2, dtype=torch.float32)

    assert isinstance(backend, PyTorchBackend)
    assert isinstance(backend.namespace, PyTorchNamespace)
    assert infer_backend(tensor) is backend
    assert backend.metadata(tensor).dtype == torch.float32
    assert backend.metadata(tensor).device == tensor.device


def test_pytorch_construction_preserves_dtype_and_device():
    backend = get_backend("pytorch")
    device = torch.device("cpu")
    original = torch.tensor([1.0, 2.0], dtype=torch.float32)

    assert backend.asarray(original) is original
    assert backend.asarray([1.0], dtype="float32").dtype == torch.float32
    assert backend.zeros(2, dtype="float32", device=device).device == device
    assert backend.ones(
        2, dtype="float64", device="cpu"
    ).dtype == torch.float64
    torch.testing.assert_close(backend.full(2, 3.0), torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(backend.eye(2), torch.eye(2))
    torch.testing.assert_close(backend.arange(3), torch.tensor([0, 1, 2]))
    torch.testing.assert_close(backend.arange(1, 4), torch.tensor([1, 2, 3]))
    torch.testing.assert_close(
        backend.linspace(0.0, 1.0, 3), torch.tensor([0.0, 0.5, 1.0])
    )
    with pytest.raises(TypeError, match="Unknown PyTorch dtype"):
        backend.zeros(2, dtype="not_a_dtype")


def test_pytorch_namespace_smoke_and_special_functions():
    namespace = get_backend("pytorch").namespace
    array = torch.tensor([[-1.0, 2.0], [3.0, 4.0]])
    positive = torch.tensor([1.0, 2.0, 3.0])

    torch.testing.assert_close(namespace.abs(array), array.abs())
    torch.testing.assert_close(
        namespace.all(array > 0, axis=1), torch.tensor([False, True])
    )
    torch.testing.assert_close(
        namespace.any(array < 0, axis=1), torch.tensor([True, False])
    )
    assert namespace.broadcast_to(positive, (2, 3)).shape == (2, 3)
    torch.testing.assert_close(
        namespace.clip(array, 0.0, 3.0),
        torch.tensor([[0.0, 2.0], [3.0, 3.0]]),
    )
    assert namespace.concatenate((array, array)).shape == (4, 2)
    torch.testing.assert_close(
        namespace.diagonal(array), torch.tensor([-1.0, 4.0])
    )
    assert namespace.expand_dims(array, axis=0).shape == (1, 2, 2)
    torch.testing.assert_close(
        namespace.isfinite(torch.tensor([1.0, torch.inf])),
        torch.tensor([True, False]),
    )
    torch.testing.assert_close(
        namespace.exp(namespace.log(positive)), positive
    )
    torch.testing.assert_close(
        namespace.logaddexp(positive, 0.0),
        torch.logaddexp(positive, torch.tensor(0.0)),
    )
    torch.testing.assert_close(
        namespace.maximum(array, 0.0),
        torch.tensor([[0.0, 2.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(
        namespace.minimum(array, 0.0),
        torch.tensor([[-1.0, 0.0], [0.0, 0.0]]),
    )
    assert namespace.mean(positive).shape == ()
    torch.testing.assert_close(
        namespace.norm(array, axis=1), torch.linalg.vector_norm(array, dim=1)
    )
    assert namespace.prod(positive).shape == ()
    torch.testing.assert_close(
        namespace.sigmoid(torch.tensor([0.0])), torch.tensor([0.5])
    )
    torch.testing.assert_close(namespace.sqrt(positive), positive.sqrt())
    torch.testing.assert_close(namespace.square(positive), positive.square())
    assert namespace.stack((array, array)).shape == (2, 2, 2)
    assert namespace.sum(positive).shape == ()
    torch.testing.assert_close(
        namespace.where(array > 0, array, 0.0),
        torch.tensor([[0.0, 2.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(
        namespace.gamma(positive), torch.tensor([1.0, 1.0, 2.0])
    )
    torch.testing.assert_close(
        namespace.gammaln(positive), torch.log(torch.tensor([1.0, 1.0, 2.0]))
    )
    assert namespace.logsumexp(torch.log(positive)).shape == ()

    with pytest.raises(BackendCapabilityError, match="arbitrary-order"):
        namespace.bessel_kv(0.5, positive)


def test_pytorch_generators_are_explicit_and_reproducible():
    backend = get_backend("pytorch")
    firstState = backend.random_state(9182)
    secondState = backend.random_state(9182)

    firstNormal, returnedState = backend.normal(
        firstState, (2, 3), dtype="float32"
    )
    secondNormal, secondState = backend.normal(
        secondState, (2, 3), dtype="float32"
    )
    assert returnedState is firstState
    torch.testing.assert_close(firstNormal, secondNormal)

    firstUniform, firstState = backend.uniform(firstState, 4)
    secondUniform, secondState = backend.uniform(secondState, 4)
    torch.testing.assert_close(firstUniform, secondUniform)

    firstPoisson, firstState = backend.poisson(firstState, 2.0, shape=4)
    secondPoisson, secondState = backend.poisson(secondState, 2.0, shape=4)
    torch.testing.assert_close(firstPoisson, secondPoisson)

    firstBinomial, _ = backend.binomial(firstState, 5, 0.4, shape=4)
    secondBinomial, _ = backend.binomial(secondState, 5, 0.4, shape=4)
    torch.testing.assert_close(firstBinomial, secondBinomial)


def test_pytorch_automatic_differentiation_contract():
    backend = get_backend("pytorch")
    x = torch.tensor([1.0, 2.0])

    def function(value):
        return backend.namespace.sum(value ** 3)

    torch.testing.assert_close(
        backend.grad(function)(x), torch.tensor([3.0, 12.0])
    )
    value, gradient = backend.value_and_grad(function)(x)
    torch.testing.assert_close(value, torch.tensor(9.0))
    torch.testing.assert_close(gradient, torch.tensor([3.0, 12.0]))
    torch.testing.assert_close(
        backend.jacobian(lambda value: value ** 2)(x),
        torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
    )
    torch.testing.assert_close(
        backend.hessian(function)(x),
        torch.tensor([[6.0, 0.0], [0.0, 12.0]]),
    )
    primal, tangent = backend.jvp(
        lambda value: value ** 2, (x,), (torch.ones_like(x),)
    )
    torch.testing.assert_close(primal, torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(tangent, torch.tensor([2.0, 4.0]))
    primal, pullback = backend.vjp(lambda value: value ** 2, x)
    torch.testing.assert_close(primal, torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(
        pullback(torch.ones_like(x))[0], torch.tensor([2.0, 4.0])
    )


def test_pytorch_compilation_vectorisation_and_control_flow():
    backend = get_backend("pytorch")
    compiled = backend.compile(
        lambda value: value * value + 1.0,
        backend="eager",
        fullgraph=True,
    )
    vectorised = backend.vectorize(lambda value: value * value)

    torch.testing.assert_close(
        compiled(torch.tensor([2.0, 3.0])), torch.tensor([5.0, 10.0])
    )
    torch.testing.assert_close(
        vectorised(torch.arange(4.0)), torch.tensor([0.0, 1.0, 4.0, 9.0])
    )

    def choose(predicate, operand):
        return backend.cond(
            predicate, lambda x: x + 1, lambda x: x - 1, operand
        )

    compiledChoose = backend.compile(
        choose, backend="eager", fullgraph=True
    )
    torch.testing.assert_close(
        compiledChoose(torch.tensor(True), torch.tensor(2.0)),
        torch.tensor(3.0),
    )


def test_pytorch_linear_algebra_and_spectral_transforms():
    backend = get_backend("pytorch")
    matrix = torch.tensor([[2.0, 0.3], [0.3, 1.1]])
    coordinate = torch.tensor([0.4, -1.2])
    chol = backend.cholesky(matrix)

    torch.testing.assert_close(chol @ chol.T, matrix)
    torch.testing.assert_close(
        backend.solve(matrix, coordinate),
        torch.tensor([0.3791469, -1.1943128]),
    )
    lowerSolution = backend.solve_triangular(chol, coordinate, lower=True)
    torch.testing.assert_close(chol @ lowerSolution, coordinate)

    array = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    torch.testing.assert_close(
        backend.dct1(array), torch.tensor([8.0, -2.0, 0.0])
    )
    torch.testing.assert_close(
        backend.dst1(array),
        torch.tensor([
            4.0 + 4.0 * 2.0 ** 0.5,
            -4.0,
            4.0 * 2.0 ** 0.5 - 4.0,
        ]),
    )
    transformGradient = torch.autograd.grad(
        backend.dct1(array).sum(), array
    )[0]
    torch.testing.assert_close(
        transformGradient, torch.tensor([3.0, 0.0, 1.0])
    )


def test_pytorch_capabilities_are_explicit():
    backend = get_backend("pytorch")
    capabilities = backend.capabilities

    assert capabilities.automaticDifferentiation
    assert capabilities.compilation
    assert capabilities.vectorisation
    assert capabilities.controlFlow
    assert capabilities.spectralTransforms
    assert not capabilities.transformedLoops

    with pytest.raises(BackendCapabilityError, match="transformed loops"):
        backend.scan(
            lambda state, value: (state + value, state),
            torch.tensor(0.0),
            torch.arange(3.0),
        )
