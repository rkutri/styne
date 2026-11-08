import numpy as np
import pytest

from styne.backend import get_backend
from styne.statistics.covariance import (
    DenseCovarianceMatrix,
    DiagonalCovarianceMatrix,
    IIDCovarianceMatrix,
    PrecisionCovarianceMatrix,
    SemidefiniteCovarianceMatrix,
)
from styne.utility.exceptions import NotPositiveDefinite


@pytest.fixture(params=["numpy", "jax", "pytorch"])
def backend(request):
    if request.param != "numpy":
        pytest.importorskip("torch" if request.param == "pytorch" else "jax")
    return get_backend(request.param)


def test_numpy_covariance_scalars_are_rank_zero_arrays():
    covariance = DenseCovarianceMatrix(
        np.array([[2.0, 0.3], [0.3, 1.1]]), scaling=1.25
    )

    assert covariance.log_determinant().shape == ()
    assert covariance.quadratic_form(np.array([0.4, -1.2])).shape == ()
    assert covariance.dual_quadratic_form(
        np.array([0.4, -1.2])
    ).shape == ()


def test_pytorch_covariances_preserve_graph_and_batches():
    torch = pytest.importorskip("torch", reason="PyTorch is optional")
    matrix = torch.tensor(
        [[2.0, 0.3], [0.3, 1.1]], dtype=torch.float64, requires_grad=True
    )
    scaling = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)
    points = torch.tensor([[0.4, -1.2], [1.0, 0.5]], dtype=torch.float64)

    covariance = DenseCovarianceMatrix(matrix, scaling)
    value = covariance.log_determinant() + covariance.dual_quadratic_form(
        points
    ).sum()
    matrixGradient, scalingGradient = torch.autograd.grad(
        value, (matrix, scaling)
    )

    assert value.shape == ()
    assert covariance.apply(points).shape == points.shape
    assert covariance.apply_inverse(points).shape == points.shape
    assert torch.isfinite(matrixGradient).all()
    assert torch.isfinite(scalingGradient)

    diagonal = DiagonalCovarianceMatrix(
        torch.tensor([1.5, 0.6], dtype=torch.float32)
    )
    iid = IIDCovarianceMatrix(2, torch.tensor(0.5, dtype=torch.float32))
    assert diagonal.to_cholesky().dtype == torch.float32
    assert iid.apply(points.to(dtype=torch.float32)).shape == points.shape


def test_jax_covariances_preserve_graph_and_batches():
    jax = pytest.importorskip("jax", reason="JAX is optional")
    jnp = pytest.importorskip("jax.numpy", reason="JAX is optional")
    matrix = jnp.array([[2.0, 0.3], [0.3, 1.1]], dtype=jnp.float32)
    points = jnp.array([[0.4, -1.2], [1.0, 0.5]], dtype=jnp.float32)

    def objective(covarianceEntries, scaling):
        covariance = DenseCovarianceMatrix(covarianceEntries, scaling)
        return covariance.log_determinant() + jnp.sum(
            covariance.dual_quadratic_form(points)
        )

    matrixGradient, scalingGradient = jax.grad(objective, argnums=(0, 1))(
        matrix, jnp.array(1.25, dtype=jnp.float32)
    )
    covariance = DenseCovarianceMatrix(matrix, jnp.array(1.25))

    assert covariance.apply(points).shape == points.shape
    assert covariance.apply_inverse(points).shape == points.shape
    assert jnp.all(jnp.isfinite(matrixGradient))
    assert jnp.isfinite(scalingGradient)

    diagonal = DiagonalCovarianceMatrix(jnp.array([1.5, 0.6]))
    iid = IIDCovarianceMatrix(2, jnp.array(0.5))
    assert diagonal.to_cholesky().dtype == jnp.float32
    assert iid.apply(points).shape == points.shape


@pytest.mark.parametrize("batchShape", ((), (3,), (2, 3)))
def test_dense_covariance_operations_match_dense_algebra(backend, batchShape):
    rng = np.random.default_rng(11)
    root = rng.standard_normal((4, 4))
    matrix = root @ root.T + 4.0 * np.eye(4)
    points = rng.standard_normal(batchShape + (4,))
    covariance = DenseCovarianceMatrix(
        backend.asarray(matrix, dtype="float32"), scaling=1.5
    )
    native = backend.asarray(points, dtype="float32")
    scaled = 1.5 * matrix
    inverse = np.linalg.inv(scaled)

    np.testing.assert_allclose(
        np.asarray(covariance.apply(native)), points @ scaled, rtol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(covariance.apply_inverse(native)), points @ inverse,
        rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(covariance.quadratic_form(native)),
        np.einsum("...i,ij,...j", points, scaled, points), rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(covariance.dual_quadratic_form(native)),
        np.einsum("...i,ij,...j", points, inverse, points), rtol=1e-4,
    )


def rank_deficient_matrix():
    root = np.random.default_rng(12).standard_normal((4, 2))
    return root @ root.T


def assert_covariance_algebra(covariance, matrix, backend):
    points = np.random.default_rng(13).standard_normal((3, 4))
    native = backend.asarray(points, dtype="float32")
    factor = np.asarray(
        covariance.to_factor() if isinstance(covariance, SemidefiniteCovarianceMatrix)
        else covariance.to_cholesky()
    )

    np.testing.assert_allclose(
        np.asarray(covariance.apply(native)), points @ matrix,
        rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(covariance.quadratic_form(native)),
        np.einsum("...i,ij,...j", points, matrix, points),
        rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(covariance.apply_chol_factor(
            covariance.apply_chol_factor_transpose(native)
        )),
        points @ matrix, rtol=1e-3, atol=1e-3,
    )
    np.testing.assert_allclose(factor @ factor.T, matrix, rtol=1e-3, atol=1e-3)


def test_semidefinite_covariance_operations_on_backends(backend):
    covariance = SemidefiniteCovarianceMatrix(
        backend.asarray(rank_deficient_matrix(), dtype="float32"), scaling=1.5
    )

    assert covariance.dimension == 4
    assert_covariance_algebra(covariance, 1.5 * rank_deficient_matrix(), backend)


def test_semidefinite_batched_square_roots(backend):
    matrices = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[2.0, 0.0], [0.0, 2.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    ], dtype=np.float32)
    covariance = SemidefiniteCovarianceMatrix(backend.asarray(matrices), scaling=1.5)

    factors = np.asarray(covariance.to_factor())

    np.testing.assert_allclose(
        factors @ np.swapaxes(factors, -1, -2), 1.5 * matrices,
        rtol=1e-5, atol=1e-6,
    )


def test_semidefinite_covariance_rejects_inverse_operations_when_singular():
    covariance = SemidefiniteCovarianceMatrix(rank_deficient_matrix())

    for operation in (covariance.apply_inverse, covariance.dual_quadratic_form):
        with pytest.raises(NotPositiveDefinite, match="singular"):
            operation(np.ones(4))
    with pytest.raises(NotPositiveDefinite, match="singular"):
        covariance.log_determinant()
    with pytest.raises(NotPositiveDefinite, match="singular"):
        covariance.to_cholesky()


def test_semidefinite_covariance_validates_spectrum_and_inverts_when_regular():
    matrix = np.array([[2.0, 0.3], [0.3, 1.1]])
    covariance = SemidefiniteCovarianceMatrix(matrix, scaling=2.0)
    vector = np.array([1.0, -1.0])

    with pytest.raises(ValueError, match="semidefinite"):
        SemidefiniteCovarianceMatrix(np.diag([1.0, -0.5]))
    with pytest.raises(ValueError, match="symmetric"):
        SemidefiniteCovarianceMatrix(np.array([[1.0, 0.2], [0.0, 1.0]]))
    np.testing.assert_allclose(
        covariance.apply_inverse(vector), np.linalg.solve(2.0 * matrix, vector)
    )
    np.testing.assert_allclose(
        covariance.dual_quadratic_form(vector),
        vector @ np.linalg.solve(2.0 * matrix, vector),
    )
    np.testing.assert_allclose(
        covariance.log_determinant(), np.log(np.linalg.det(2.0 * matrix))
    )
    np.testing.assert_allclose(
        covariance.with_scaling(1.0).apply(vector), matrix @ vector
    )
    factor = covariance.to_cholesky()
    np.testing.assert_allclose(factor, np.tril(factor))
    np.testing.assert_allclose(factor @ factor.T, 2.0 * matrix)


@pytest.mark.parametrize("covarianceType", [
    DenseCovarianceMatrix, PrecisionCovarianceMatrix,
])
@pytest.mark.parametrize("pointShape", [(2,), (2, 2), (3, 2, 2)])
def test_covariance_solves_with_different_batched_matrices(
        backend, covarianceType, pointShape):
    matrices = np.stack([np.eye(2), 2.0 * np.eye(2)])
    points = np.ones(pointShape, dtype=np.float32)
    covariance = covarianceType(backend.asarray(matrices, dtype="float32"))
    native = backend.asarray(points)
    expectedInverse = np.broadcast_to(
        np.array([[1.0, 1.0], [0.5, 0.5]]),
        np.broadcast_shapes(pointShape, (2, 2)),
    )
    if covarianceType is DenseCovarianceMatrix:
        actual = covariance.apply_inverse(native)
        quadratic = covariance.dual_quadratic_form(native)
    else:
        actual = covariance.apply(native)
        quadratic = covariance.quadratic_form(native)
        factors = np.asarray(covariance.to_cholesky())
        np.testing.assert_allclose(
            factors @ np.swapaxes(factors, -1, -2),
            np.linalg.inv(matrices), rtol=1e-5, atol=1e-6,
        )
    np.testing.assert_allclose(actual, expectedInverse, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(quadratic, expectedInverse.sum(axis=-1), rtol=1e-5)


def test_precision_covariance_operations_on_backends(backend):
    root = np.random.default_rng(14).standard_normal((4, 4))
    precision = root @ root.T + 4.0 * np.eye(4)
    matrix = 1.5 * np.linalg.inv(precision)
    covariance = PrecisionCovarianceMatrix(
        backend.asarray(precision, dtype="float32"), scaling=1.5
    )
    points = np.random.default_rng(15).standard_normal((3, 4))
    native = backend.asarray(points, dtype="float32")

    assert_covariance_algebra(covariance, matrix, backend)
    np.testing.assert_allclose(
        np.asarray(covariance.apply_inverse(native)), points @ precision / 1.5,
        rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(covariance.dual_quadratic_form(native)),
        np.einsum("...i,ij,...j", points, precision, points) / 1.5,
        rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(covariance.log_determinant()), np.log(np.linalg.det(matrix)),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(covariance.precision), precision, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(covariance.with_scaling(1.0).apply(native)),
        points @ np.linalg.inv(precision), rtol=1e-4, atol=1e-4,
    )


def test_precision_covariance_requires_positive_definite_precision():
    with pytest.raises(NotPositiveDefinite):
        PrecisionCovarianceMatrix(rank_deficient_matrix())


@pytest.mark.parametrize("precisionType", [
    DenseCovarianceMatrix, DiagonalCovarianceMatrix, SemidefiniteCovarianceMatrix,
])
def test_precision_covariance_accepts_a_scaled_operator(backend, precisionType):
    matrix = np.array([[2.0, 0.3], [0.3, 1.1]], dtype=np.float32)
    if precisionType is DiagonalCovarianceMatrix:
        matrix = np.diag(np.diag(matrix))
        values = np.diag(matrix).copy()
    else:
        values = matrix
    operator = precisionType(backend.asarray(values), scaling=2.5)
    covariance = PrecisionCovarianceMatrix(operator, scaling=1.7)
    vector = backend.asarray([1.0, -0.5], dtype="float32")
    expected = 1.7 * np.linalg.inv(2.5 * matrix)

    assert np.asarray(covariance.scaling) == pytest.approx(1.7)
    np.testing.assert_allclose(covariance.precision, 2.5 * matrix, rtol=1e-5)
    factor = np.asarray(covariance.precisionFactor)
    np.testing.assert_allclose(factor @ factor.T, 2.5 * matrix, rtol=1e-5)
    np.testing.assert_allclose(covariance.apply(vector), expected @ [1.0, -0.5],
                               rtol=1e-5)
    np.testing.assert_allclose(covariance.apply_inverse(vector),
                               np.linalg.solve(expected, [1.0, -0.5]), rtol=1e-5)
    np.testing.assert_allclose(covariance.log_determinant(),
                               np.linalg.slogdet(expected)[1], rtol=1e-5)
    if precisionType is DenseCovarianceMatrix:
        assert covariance._precisionFactor is operator._cholFactor


@pytest.mark.parametrize("covarianceType", [
    DenseCovarianceMatrix, PrecisionCovarianceMatrix,
])
def test_batched_triangular_solves_compile_and_differentiate(covarianceType):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    matrices = jnp.stack([jnp.eye(2), 2.0 * jnp.eye(2)])
    points = jnp.ones((3, 2, 2))

    def solve(entries, vectors):
        covariance = covarianceType(entries).with_scaling(1.5)
        if covarianceType is DenseCovarianceMatrix:
            return covariance.apply_inverse(vectors)
        return covariance.apply(vectors)

    result = jax.jit(solve)(matrices, points)
    expected = np.ones((3, 2, 2)) * np.array([1.0, 0.5])[None, :, None]
    expected *= 1.5 if covarianceType is PrecisionCovarianceMatrix else 1.0 / 1.5
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    gradient = jax.grad(lambda entries: jnp.sum(solve(entries, points)))(matrices)
    assert jnp.all(jnp.isfinite(gradient))


def test_batched_numpy_solve_supports_scipy_without_batch_api(monkeypatch):
    from scipy import linalg

    solve = linalg.solve_triangular
    calls = []

    def single_matrix_only(matrix, rightHandSide, *, lower):
        assert matrix.ndim == 2
        calls.append(matrix.shape)
        return solve(matrix, rightHandSide, lower=lower)

    monkeypatch.setattr(linalg, "solve_triangular", single_matrix_only)
    covariance = DenseCovarianceMatrix(np.stack([np.eye(2), 2.0 * np.eye(2)]))

    np.testing.assert_allclose(
        covariance.apply_inverse(np.ones((2, 2))), [[1.0, 1.0], [0.5, 0.5]]
    )
    assert len(calls) == 4
