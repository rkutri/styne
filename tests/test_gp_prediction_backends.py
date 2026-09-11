import numpy as np
import pytest

from styne.backend import infer_backend
from styne.gp.dna import DNAFourierExpansion
from styne.gp.gaussianprocess import GaussianProcess
from styne.model.representation.bspline import BSpline1D
from styne.model.sglmm import SGLMM
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import Grid, UniformGrid
from styne.utility.interpolation import (
    bilinear_interpolation_matrix,
    linear_interpolation_matrix,
)


def small_process(representation, covariance):
    if representation == "direct":
        return GaussianProcess.direct(
            UniformGrid(0.0, 1.0, 6), covariance
        )
    return GaussianProcess.bspline(
        covariance, BSpline1D(6, degree=3, boundary=[0.0, 1.0])
    )


@pytest.mark.parametrize("representation", ["direct", "bspline"])
def test_jax_gp_evaluation_and_sampling_preserve_backend_and_graph(
        representation):
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    logLengthScale = jnp.asarray(-0.7)
    process = small_process(
        representation,
        MaternCovariance1D(
            jnp.exp(logLengthScale), 1.5, jnp.asarray(1.2)
        ),
    )
    sites = UniformGrid(0.1, 0.9, 5)
    coefficient = jnp.linspace(
        -0.5, 0.8, process.parameterDimension
    )
    evaluation = process.bind(sites)
    evaluate = jax.jit(evaluation.evaluate)

    values = evaluate(coefficient)
    batchedValues = evaluate(jnp.stack((coefficient, -coefficient)))
    coefficientGradient = jax.grad(
        lambda current: jnp.sum(evaluate(current))
    )(coefficient)
    field, _ = process.sampler.sample(jax.random.key(11))
    sampleValues = field.evaluate(sites)
    hyperGradient = jax.grad(
        lambda logLength: jnp.sum(
            small_process(
                representation,
                MaternCovariance1D(
                    jnp.exp(logLength), 1.5, jnp.asarray(1.2)
                ),
            ).sampler.sample(jax.random.key(12))[0].evaluate(sites)
        )
    )(logLengthScale)

    assert infer_backend(values).name == "jax"
    assert values.dtype == coefficient.dtype
    assert values.shape == (5,)
    assert batchedValues.shape == (2, 5)
    assert coefficientGradient.shape == coefficient.shape
    assert sampleValues.shape == (5,)
    assert jnp.isfinite(hyperGradient)


@pytest.mark.parametrize("representation", ["direct", "bspline"])
def test_pytorch_gp_evaluation_and_sampling_preserve_device_dtype_and_graph(
        representation):
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    logLengthScale = torch.tensor(
        -0.7, dtype=torch.float64, requires_grad=True
    )
    process = small_process(
        representation,
        MaternCovariance1D(
            torch.exp(logLengthScale), 1.5,
            torch.tensor(1.2, dtype=torch.float64),
        ),
    )
    sites = UniformGrid(0.1, 0.9, 5)
    coefficient = torch.linspace(
        -0.5, 0.8, process.parameterDimension,
        dtype=torch.float64, requires_grad=True,
    )

    values = process.evaluate(coefficient, sites)
    batchedValues = process.evaluate(
        torch.stack((coefficient, -coefficient)), sites
    )
    field, _ = process.sampler.sample(
        torch.Generator().manual_seed(11)
    )
    sampleValues = field.evaluate(sites)
    coefficientGradient, hyperGradient = torch.autograd.grad(
        values.sum() + sampleValues.sum(),
        (coefficient, logLengthScale),
    )

    assert infer_backend(values).name == "pytorch"
    assert values.device == coefficient.device
    assert values.dtype == coefficient.dtype
    assert values.shape == (5,)
    assert batchedValues.shape == (2, 5)
    assert coefficientGradient.shape == coefficient.shape
    assert sampleValues.shape == (5,)
    assert torch.isfinite(hyperGradient)


def test_jax_high_resolution_bspline_sample_is_finite():
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    covariance = MaternCovariance1D(
        jnp.asarray(0.2, dtype=jnp.float32), 1.5,
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    process = GaussianProcess.bspline(
        covariance, BSpline1D(100, degree=3, boundary=[0.0, 1.0])
    )

    field, _ = process.sampler.sample(jax.random.key(11))

    assert field.coordinate.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(field.coordinate))


def test_pytorch_high_resolution_bspline_sample_is_finite():
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    covariance = MaternCovariance1D(
        torch.tensor(0.2, dtype=torch.float32), 1.5,
        torch.tensor(1.0, dtype=torch.float32),
    )
    process = GaussianProcess.bspline(
        covariance, BSpline1D(100, degree=3, boundary=[0.0, 1.0])
    )

    field, _ = process.sampler.sample(torch.Generator().manual_seed(11))

    assert field.coordinate.dtype == torch.float32
    assert torch.all(torch.isfinite(field.coordinate))


def test_jax_sglmm_prediction_returns_a_differentiable_array():
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    process = GaussianProcess.dna(
        MaternCovariance1D(
            jnp.asarray(0.4), 1.5, jnp.asarray(1.2)
        ),
        q=4,
        d=1,
    )
    model = SGLMM(
        process, UniformGrid(0.0, 1.0, 8),
        features=np.zeros((8, 2)),
    )
    sites = UniformGrid(0.1, 0.9, 5)
    features = np.arange(10, dtype=float).reshape(5, 2)
    latent = jnp.linspace(-0.5, 0.8, process.parameterDimension)
    fixed = jnp.asarray([0.4, -0.2])
    predict = jax.jit(
        lambda z, beta: model.predict(
            (z, beta), sites, features=features
        )
    )

    values = predict(latent, fixed)
    latentGradient, fixedGradient = jax.grad(
        lambda z, beta: jnp.sum(predict(z, beta)), argnums=(0, 1)
    )(latent, fixed)

    assert infer_backend(values).name == "jax"
    assert values.shape == (5,)
    assert latentGradient.shape == latent.shape
    assert fixedGradient.shape == fixed.shape


def test_pytorch_sglmm_prediction_returns_a_differentiable_array():
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    process = GaussianProcess.dna(
        MaternCovariance1D(
            torch.tensor(0.4, dtype=torch.float64), 1.5,
            torch.tensor(1.2, dtype=torch.float64),
        ),
        q=4,
        d=1,
    )
    model = SGLMM(
        process, UniformGrid(0.0, 1.0, 8),
        features=np.zeros((8, 2)),
    )
    sites = UniformGrid(0.1, 0.9, 5)
    features = np.arange(10, dtype=float).reshape(5, 2)
    latent = torch.linspace(
        -0.5, 0.8, process.parameterDimension,
        dtype=torch.float64, requires_grad=True,
    )
    fixed = torch.tensor(
        [0.4, -0.2], dtype=torch.float64, requires_grad=True
    )

    values = model.predict((latent, fixed), sites, features=features)
    latentGradient, fixedGradient = torch.autograd.grad(
        values.sum(), (latent, fixed)
    )

    assert infer_backend(values).name == "pytorch"
    assert values.device == latent.device
    assert values.dtype == latent.dtype
    assert values.shape == (5,)
    assert latentGradient.shape == latent.shape
    assert fixedGradient.shape == fixed.shape


def test_dna_native_grid_bypasses_interpolation():
    expansion = DNAFourierExpansion((3, 2), d=2)
    evaluation = expansion.bind(expansion.nativeGrid)
    coefficient = np.linspace(-0.4, 0.8, expansion.dimension)

    assert evaluation._native
    assert not hasattr(evaluation, "_indices")
    np.testing.assert_allclose(
        evaluation.evaluate(coefficient),
        expansion.evaluate_native(coefficient),
    )


def test_dna_sparse_arbitrary_site_interpolation_matches_dense_reference():
    expansion = DNAFourierExpansion((3, 2), d=2)
    sites = Grid(np.array([[0.1, 0.2], [0.4, 0.7], [0.9, 0.6]]))
    evaluation = expansion.bind(sites)
    coefficient = np.linspace(
        -0.8, 0.7, 2 * expansion.dimension
    ).reshape(2, expansion.dimension)
    axis0 = np.linspace(0.0, 1.0, 5)
    axis1 = np.linspace(0.0, 1.0, 4)
    dense = bilinear_interpolation_matrix(
        sites.to_array(), axis0, axis1
    ).toarray()

    assert not hasattr(evaluation, "_interpolation")
    assert evaluation._indices.shape == (len(sites), 4)
    np.testing.assert_allclose(
        evaluation.evaluate(coefficient),
        expansion.evaluate_native(coefficient) @ dense.T,
    )


def test_dna_sparse_1d_interpolation_matches_dense_reference():
    expansion = DNAFourierExpansion(4, d=1)
    sites = UniformGrid(0.1, 0.9, 5)
    evaluation = expansion.bind(sites)
    coefficient = np.linspace(-0.8, 0.7, expansion.dimension)
    dense = linear_interpolation_matrix(
        sites.to_array().ravel(), np.linspace(0.0, 1.0, 6)
    ).toarray()

    assert evaluation._indices.shape == (len(sites), 2)
    np.testing.assert_allclose(
        evaluation.evaluate(coefficient),
        expansion.evaluate_native(coefficient) @ dense.T,
    )


def test_jax_dna_prediction_and_sampling_preserve_graph_and_key():
    jax = pytest.importorskip("jax", reason="JAX is an optional backend")
    jnp = pytest.importorskip("jax.numpy")
    logLengthScale = jnp.asarray(-0.7)
    covariance = MaternCovariance1D(jnp.exp(logLengthScale), 1.5, 1.2)
    gp = GaussianProcess.dna(covariance, q=4, d=1)
    sites = UniformGrid(0.1, 0.9, 5)
    coefficient = jnp.linspace(-0.5, 0.8, gp.parameterDimension)

    predict = jax.jit(lambda current: gp.evaluate(current, sites))
    values = predict(coefficient)
    gradient = jax.grad(lambda current: jnp.sum(predict(current)))(coefficient)
    batchedValues = predict(jnp.stack((coefficient, -coefficient)))
    field, nextKey = gp.sampler.sample(jax.random.key(12))

    assert values.shape == (5,)
    assert gradient.shape == coefficient.shape
    assert batchedValues.shape == (2, 5)
    assert jnp.all(jnp.isfinite(gradient))
    assert field.coordinate.shape == coefficient.shape
    assert not jnp.array_equal(nextKey, jax.random.key(12))
    assert jnp.isfinite(jax.grad(
        lambda logLength: jnp.sum(
            GaussianProcess.dna(
                MaternCovariance1D(jnp.exp(logLength), 1.5, 1.2), q=4, d=1
            ).sampler.sample(jax.random.key(13))[0].evaluate(sites)
        )
    )(logLengthScale))


def test_pytorch_dna_prediction_and_sampling_preserve_graph():
    torch = pytest.importorskip(
        "torch", reason="PyTorch is an optional backend"
    )
    logLengthScale = torch.tensor(
        -0.7, dtype=torch.float64, requires_grad=True
    )
    covariance = MaternCovariance1D(torch.exp(logLengthScale), 1.5, 1.2)
    gp = GaussianProcess.dna(covariance, q=4, d=1)
    sites = UniformGrid(0.1, 0.9, 5)
    coefficient = torch.linspace(
        -0.5, 0.8, gp.parameterDimension, dtype=torch.float64,
        requires_grad=True,
    )

    values = gp.evaluate(coefficient, sites)
    coefficientGradient, = torch.autograd.grad(values.sum(), coefficient)
    batchedCoefficient = torch.stack(
        (coefficient.detach(), -coefficient.detach())
    )
    batchedValues = gp.evaluate(batchedCoefficient, sites)
    generator = torch.Generator().manual_seed(12)
    field, nextState = gp.sampler.sample(generator)
    sampleGradient, = torch.autograd.grad(
        field.evaluate(sites).sum(), logLengthScale
    )

    assert values.shape == (5,)
    assert coefficientGradient.shape == coefficient.shape
    assert batchedValues.shape == (2, 5)
    assert field.coordinate.shape == coefficient.shape
    assert nextState is generator
    assert torch.isfinite(sampleGradient)
