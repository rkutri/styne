import numpy as np
import pytest

from styne.backend import BackendCapabilityError
from styne.gp.dna import DNAFourierComponentExpansion, DNAFourierExpansion
from styne.gp.gaussianprocess import GaussianProcess
from styne.gp.dnautility import BC, BoundaryCondition
from styne.statistics.stationary import MaternCovariance1D
from tests.reference_oracles import (
    _axis_synthesis_matrix,
    compute_log_length_multiplier,
    dna_adjoint_synthesis,
    dna_synthesis_matrix,
)


def _boundary(*conditions):
    return BoundaryCondition(list(conditions))


@pytest.mark.parametrize("q", [1, 3, 6])
@pytest.mark.parametrize("condition", [BC.NEUMANN, BC.DIRICHLET])
def test_1d_component_matches_021_type_one_transform(q, condition):
    """Pin the 0.2.1 DCT-I/DST-I extension and normalization exactly."""
    expansion = DNAFourierComponentExpansion(_boundary(condition), q)
    coefficient = np.linspace(-0.7, 1.1, expansion.dimension)

    expected = _axis_synthesis_matrix(q, condition) @ coefficient
    np.testing.assert_allclose(
        expansion.evaluate_native(coefficient), expected,
        rtol=2e-14, atol=2e-14,
    )


def test_neumann_padding_preserves_type_one_endpoint_values():
    """The last padded coefficient is zero; a DCT-II gives different ends."""
    q = 5
    expansion = DNAFourierComponentExpansion(_boundary(BC.NEUMANN), q)
    coefficient = np.zeros(q + 1)
    coefficient[-1] = 1.0

    value = expansion.evaluate_native(coefficient)
    expected = np.sqrt(2.0) * np.cos(
        np.pi * q * np.arange(q + 2) / (q + 1)
    )
    np.testing.assert_allclose(value, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(value[[0, -1]], [np.sqrt(2.0), -np.sqrt(2.0)])


def test_dirichlet_padding_restores_zero_boundary_values():
    q = 5
    expansion = DNAFourierComponentExpansion(_boundary(BC.DIRICHLET), q)
    coefficient = np.linspace(0.2, 1.0, q)
    value = expansion.evaluate_native(coefficient)

    assert value.shape == (q + 2,)
    np.testing.assert_array_equal(value[[0, -1]], 0.0)
    np.testing.assert_allclose(
        value, _axis_synthesis_matrix(q, BC.DIRICHLET) @ coefficient,
        rtol=2e-14, atol=2e-14,
    )


@pytest.mark.parametrize(
    "conditions",
    [
        (BC.NEUMANN, BC.NEUMANN),
        (BC.NEUMANN, BC.DIRICHLET),
        (BC.DIRICHLET, BC.NEUMANN),
        (BC.DIRICHLET, BC.DIRICHLET),
    ],
)
def test_rectangular_2d_components_match_021_tensor_product(conditions):
    q = (2, 4)
    expansion = DNAFourierComponentExpansion(_boundary(*conditions), q)
    coefficient = np.linspace(-0.4, 0.9, expansion.dimension)
    basis = np.kron(
        _axis_synthesis_matrix(q[0], conditions[0]),
        _axis_synthesis_matrix(q[1], conditions[1]),
    )

    np.testing.assert_allclose(
        expansion.evaluate_native(coefficient), basis @ coefficient,
        rtol=3e-14, atol=3e-14,
    )


@pytest.mark.parametrize("q,d", [(4, 1), ((2, 3), 2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_full_batched_synthesis_matches_retained_021_matrix(q, d, dtype):
    expansion = DNAFourierExpansion(q, d=d)
    dimension = expansion.dimension
    weights = np.linspace(0.3, 1.4, dimension, dtype=dtype)
    coefficient = np.linspace(-0.8, 0.7, 3 * dimension, dtype=dtype).reshape(
        3, dimension
    )
    expansion = expansion.with_spectral_weights(weights)
    reference = dna_synthesis_matrix(q, d).astype(dtype)

    actual = expansion.evaluate_native(coefficient)
    expected = (coefficient * weights) @ reference.T
    tolerance = 3e-5 if dtype == np.float32 else 5e-14
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance
    )
    assert actual.dtype == dtype


@pytest.mark.parametrize("q,d", [(4, 1), ((2, 3), 2)])
@pytest.mark.parametrize("dtypeName", ["float32", "float64"])
def test_jax_batched_values_match_021_for_both_dtypes(q, d, dtypeName):
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    with jax.enable_x64(dtypeName == "float64"):
        dtype = getattr(jnp, dtypeName)
        matrix = jnp.asarray(dna_synthesis_matrix(q, d), dtype=dtype)
        dimension = matrix.shape[1]
        coefficient = jnp.linspace(
            -0.8, 0.7, 2 * dimension, dtype=dtype
        ).reshape(2, dimension)
        weights = jnp.linspace(0.3, 1.4, dimension, dtype=dtype)
        expansion = DNAFourierExpansion(q, d=d, spectralWeights=weights)

        actual = jax.jit(expansion.evaluate_native)(coefficient)
        expected = (coefficient * weights) @ matrix.T
        tolerance = 4e-5 if dtypeName == "float32" else 8e-14
        np.testing.assert_allclose(
            actual, expected, rtol=tolerance, atol=tolerance
        )
        assert actual.dtype == dtype


@pytest.mark.parametrize("q,d", [(4, 1), ((2, 3), 2)])
@pytest.mark.parametrize("dtypeName", ["float32", "float64"])
def test_pytorch_batched_values_match_021_for_both_dtypes(q, d, dtypeName):
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    dtype = getattr(torch, dtypeName)
    matrix = torch.as_tensor(dna_synthesis_matrix(q, d), dtype=dtype)
    dimension = matrix.shape[1]
    coefficient = torch.linspace(
        -0.8, 0.7, 2 * dimension, dtype=dtype
    ).reshape(2, dimension)
    weights = torch.linspace(0.3, 1.4, dimension, dtype=dtype)
    expansion = DNAFourierExpansion(q, d=d, spectralWeights=weights)

    actual = expansion.evaluate_native(coefficient)
    expected = (coefficient * weights) @ matrix.T
    tolerance = 4e-5 if dtypeName == "float32" else 8e-14
    torch.testing.assert_close(
        actual, expected, rtol=tolerance, atol=tolerance
    )
    assert actual.dtype == dtype


@pytest.mark.parametrize("q,d", [(4, 1), ((2, 3), 2)])
def test_jax_jit_values_and_weight_gradients_match_021_matrix(q, d):
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    dimension = dna_synthesis_matrix(q, d).shape[1]
    coefficient = jnp.linspace(-0.8, 0.7, dimension)
    weights = jnp.linspace(0.3, 1.4, dimension)
    cotangent = jnp.linspace(-0.5, 0.6, dna_synthesis_matrix(q, d).shape[0])

    def objective(currentWeights):
        expansion = DNAFourierExpansion(q, d=d, spectralWeights=currentWeights)
        return jnp.vdot(expansion.evaluate_native(coefficient), cotangent)

    value, gradient = jax.jit(jax.value_and_grad(objective))(weights)
    matrix = jnp.asarray(dna_synthesis_matrix(q, d), dtype=weights.dtype)
    expectedValue = jnp.vdot(matrix @ (weights * coefficient), cotangent)
    expectedGradient = coefficient * (matrix.T @ cotangent)
    np.testing.assert_allclose(value, expectedValue, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        gradient, expectedGradient, rtol=3e-5, atol=3e-6
    )


@pytest.mark.parametrize("q,d", [(4, 1), ((2, 3), 2)])
def test_pytorch_values_and_weight_gradients_match_021_matrix(q, d):
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    dimension = dna_synthesis_matrix(q, d).shape[1]
    coefficient = torch.linspace(-0.8, 0.7, dimension, dtype=torch.float64)
    weights = torch.linspace(
        0.3, 1.4, dimension, dtype=torch.float64, requires_grad=True
    )
    cotangent = torch.linspace(
        -0.5, 0.6, dna_synthesis_matrix(q, d).shape[0], dtype=torch.float64
    )
    expansion = DNAFourierExpansion(q, d=d, spectralWeights=weights)
    value = torch.vdot(expansion.evaluate_native(coefficient), cotangent)
    gradient, = torch.autograd.grad(value, weights)

    matrix = torch.as_tensor(dna_synthesis_matrix(q, d), dtype=torch.float64)
    torch.testing.assert_close(
        value, torch.vdot(matrix @ (weights * coefficient), cotangent)
    )
    torch.testing.assert_close(gradient, coefficient * (matrix.T @ cotangent))


@pytest.mark.parametrize("q,d", [(4, 1), ((2, 3), 2)])
def test_jax_synthesis_vjp_matches_retained_021_oracle(q, d):
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    expansion = DNAFourierExpansion(q, d=d)
    coefficient = jnp.linspace(-0.8, 0.7, expansion.dimension)
    cotangent = jnp.linspace(
        -0.5, 0.6, dna_synthesis_matrix(q, d).shape[0]
    )

    gradient = jax.grad(
        lambda current: jnp.vdot(
            expansion.evaluate_native(current), cotangent
        )
    )(coefficient)

    expected = dna_adjoint_synthesis(q, d, cotangent)
    np.testing.assert_allclose(gradient, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("q,d", [(4, 1), ((2, 3), 2)])
def test_pytorch_synthesis_vjp_matches_retained_021_oracle(q, d):
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    expansion = DNAFourierExpansion(q, d=d)
    coefficient = torch.linspace(
        -0.8, 0.7, expansion.dimension, dtype=torch.float64,
        requires_grad=True,
    )
    cotangent = torch.linspace(
        -0.5, 0.6, dna_synthesis_matrix(q, d).shape[0], dtype=torch.float64,
    )

    value = torch.vdot(expansion.evaluate_native(coefficient), cotangent)
    gradient, = torch.autograd.grad(value, coefficient)

    expected = torch.as_tensor(
        dna_adjoint_synthesis(q, d, cotangent), dtype=torch.float64
    )
    torch.testing.assert_close(gradient, expected)


def test_numpy_dna_exposes_no_runtime_handwritten_adjoint():
    expansion = DNAFourierExpansion(4, d=1)
    evaluation = expansion.bind(np.linspace(0.1, 0.9, 5))

    assert not hasattr(expansion, "adjoint_synthesis")
    with pytest.raises(BackendCapabilityError, match="no NumPy adjoint"):
        evaluation.adjoint_derivative(
            np.zeros(expansion.dimension), np.ones(5)
        )


def test_jax_dna_hypergradient_matches_analytic_multiplier_oracle():
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    q = 4
    coefficient = jnp.linspace(-0.6, 0.8, 2 * q + 1)

    def objective(logLengthScale, logSigma):
        covariance = MaternCovariance1D(
            jnp.exp(logLengthScale), 1.5, jnp.exp(2.0 * logSigma)
        )
        gp = GaussianProcess.dna(covariance, q=q, d=1)
        return jnp.sum(gp.expansion.evaluate_native(coefficient))

    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        jnp.asarray(-0.7), jnp.asarray(0.2)
    )
    lengthScale = np.exp(-0.7)
    covariance = MaternCovariance1D(lengthScale, 1.5, np.exp(0.4))
    weights = np.asarray(
        GaussianProcess.dna(covariance, q=q, d=1).expansion.spectralWeights
    )
    multiplier = compute_log_length_multiplier(q, 1.0, 1, 1.5, lengthScale)
    matrix = dna_synthesis_matrix(q, 1)
    expected = np.sum(
        (matrix.T @ np.ones(matrix.shape[0])) * coefficient * weights
        * multiplier
    )

    assert jnp.isfinite(value)
    assert all(jnp.isfinite(gradient) for gradient in gradients)
    np.testing.assert_allclose(gradients[0], expected, rtol=3e-5, atol=3e-6)
    assert jnp.abs(gradients[1]) > 1e-7


def test_pytorch_dna_hypergradient_matches_analytic_multiplier_oracle():
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    q = 4
    logLengthScale = torch.tensor(
        -0.7, dtype=torch.float64, requires_grad=True
    )
    logSigma = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    coefficient = torch.linspace(-0.6, 0.8, 2 * q + 1, dtype=torch.float64)
    covariance = MaternCovariance1D(
        torch.exp(logLengthScale), 1.5, torch.exp(2.0 * logSigma)
    )
    gp = GaussianProcess.dna(covariance, q=q, d=1)
    value = gp.expansion.evaluate_native(coefficient).sum()
    gradients = torch.autograd.grad(value, (logLengthScale, logSigma))
    lengthScale = float(torch.exp(logLengthScale).detach())
    marginalVariance = float(torch.exp(2.0 * logSigma).detach())
    referenceCovariance = MaternCovariance1D(
        lengthScale, 1.5, marginalVariance
    )
    weights = np.asarray(
        GaussianProcess.dna(
            referenceCovariance, q=q, d=1
        ).expansion.spectralWeights
    )
    multiplier = compute_log_length_multiplier(q, 1.0, 1, 1.5, lengthScale)
    matrix = dna_synthesis_matrix(q, 1)
    expected = np.sum(
        (matrix.T @ np.ones(matrix.shape[0])) * coefficient.detach().numpy()
        * weights * multiplier
    )

    assert torch.isfinite(value)
    assert all(torch.isfinite(gradient) for gradient in gradients)
    torch.testing.assert_close(gradients[0], torch.as_tensor(expected))
    assert torch.abs(gradients[1]) > 1e-7
