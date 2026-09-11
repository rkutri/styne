import numpy as np
import pytest

from styne.gp.gaussianprocess import GaussianProcess
from styne.model import DifferentiableForwardMap
from styne.model.sglmm import SGLMM
from styne.model.trend import ConstantTrend
from styne.parameter.block import BlockParameter
from styne.parameter.vector import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.data import Data
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.response import (
    BinomialResponse,
    GaussianResponse,
    PoissonResponse,
)
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import Grid


def make_model(features):
    sites = Grid(np.array([[0.1], [0.35], [0.8]]))
    gp = GaussianProcess.dna(
        MaternCovariance1D(0.25, 1.5, 1.0), q=4, d=1
    )
    return (
        SGLMM(gp, sites, features=features, trend=ConstantTrend(0.4)),
        gp,
        sites,
    )


def response_and_measurement(name, array, nSites):
    if name == "gaussian":
        return (
            GaussianResponse(IIDCovarianceMatrix(nSites, array(0.5))),
            array([[0.8], [-0.3], [1.2]]),
        )
    if name == "poisson":
        return PoissonResponse(), array([[2.0], [1.0], [3.0]])
    return BinomialResponse(4), array([[2.0], [1.0], [3.0]])


@pytest.mark.parametrize("responseName", ["gaussian", "poisson", "binomial"])
def test_numpy_sglmm_value_matches_explicit_components(responseName):
    features = np.array([[1.0, -0.5], [0.2, 0.3], [-0.4, 0.7]])
    model, gp, sites = make_model(features)
    latent = np.linspace(-0.2, 0.3, gp.parameterDimension)
    fixedEffect = np.array([0.1, -0.2])
    parameter = BlockParameter([Vector(latent), Vector(fixedEffect)])
    response, measurement = response_and_measurement(
        responseName, np.asarray, len(sites)
    )
    data = Data(1, sites.to_array())
    data.measurement = measurement

    expected = gp.evaluate(latent, sites) + fixedEffect @ features.T + 0.4
    evaluation = model(parameter)
    likelihood = RegressionLikelihood(data, model, response)

    np.testing.assert_allclose(evaluation, expected)
    assert np.isfinite(likelihood.evaluate_log(parameter))
    assert isinstance(model, DifferentiableForwardMap)
    assert np.all(np.isfinite(
        likelihood.evaluate_log_gradient(parameter)
    ))


@pytest.mark.parametrize("responseName", ["gaussian", "poisson", "binomial"])
def test_jax_sglmm_likelihood_gradient_matches_fixed_effect_score(
        responseName):
    jax = pytest.importorskip("jax", reason="JAX is optional")
    jnp = pytest.importorskip("jax.numpy", reason="JAX is optional")

    features = jnp.array([[1.0, -0.5], [0.2, 0.3], [-0.4, 0.7]])
    model, gp, sites = make_model(features)
    response, measurement = response_and_measurement(
        responseName, jnp.asarray, len(sites)
    )
    data = Data(1, sites.to_array())
    data.measurement = measurement
    likelihood = RegressionLikelihood(data, model, response)
    latent = jnp.linspace(-0.2, 0.3, gp.parameterDimension)
    fixedEffect = jnp.array([0.1, -0.2])

    def log_likelihood(latentCoordinate, fixedCoordinate):
        return likelihood.evaluate_log(BlockParameter([
            Vector(latentCoordinate), Vector(fixedCoordinate)
        ]))

    value, (latentGradient, fixedGradient) = jax.jit(
        jax.value_and_grad(log_likelihood, argnums=(0, 1))
    )(latent, fixedEffect)
    eta = model(BlockParameter([Vector(latent), Vector(fixedEffect)]))
    y = measurement.reshape((-1,))
    if responseName == "gaussian":
        score = (y - eta) / 0.5
    elif responseName == "poisson":
        score = y - jnp.exp(eta)
    else:
        score = y - 4.0 * jax.nn.sigmoid(eta)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(latentGradient))
    np.testing.assert_allclose(
        fixedGradient, features.T @ score, rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize("responseName", ["gaussian", "poisson", "binomial"])
def test_pytorch_sglmm_likelihood_gradient_matches_fixed_effect_score(
        responseName):
    torch = pytest.importorskip("torch", reason="PyTorch is optional")

    features = torch.tensor(
        [[1.0, -0.5], [0.2, 0.3], [-0.4, 0.7]], dtype=torch.float64
    )
    model, gp, sites = make_model(features)
    response, measurement = response_and_measurement(
        responseName, torch.as_tensor, len(sites)
    )
    data = Data(1, sites.to_array())
    data.measurement = measurement.to(dtype=torch.float64)
    likelihood = RegressionLikelihood(data, model, response)
    latent = torch.linspace(
        -0.2, 0.3, gp.parameterDimension, dtype=torch.float64,
        requires_grad=True,
    )
    fixedEffect = torch.tensor(
        [0.1, -0.2], dtype=torch.float64, requires_grad=True
    )
    parameter = BlockParameter([Vector(latent), Vector(fixedEffect)])

    value = likelihood.evaluate_log(parameter)
    latentGradient, fixedGradient = torch.autograd.grad(
        value, (latent, fixedEffect)
    )
    eta = model(parameter)
    y = data.measurement.reshape((-1,))
    if responseName == "gaussian":
        score = (y - eta) / 0.5
    elif responseName == "poisson":
        score = y - torch.exp(eta)
    else:
        score = y - 4.0 * torch.sigmoid(eta)

    assert torch.isfinite(value)
    assert torch.isfinite(latentGradient).all()
    torch.testing.assert_close(fixedGradient, features.T @ score)


def test_sglmm_prediction_uses_backend_native_query_features():
    torch = pytest.importorskip("torch", reason="PyTorch is optional")

    features = torch.tensor([[1.0], [0.5], [-0.2]], dtype=torch.float64)
    model, gp, _ = make_model(features)
    queryGrid = Grid(np.array([[0.2], [0.6]]))
    queryFeatures = torch.tensor([[0.3], [-0.4]], dtype=torch.float64)
    latent = torch.linspace(
        -0.2, 0.3, gp.parameterDimension, dtype=torch.float64
    )
    fixedEffect = torch.tensor([0.7], dtype=torch.float64)

    prediction = model.predict(
        model.prepare(BlockParameter([Vector(latent), Vector(fixedEffect)])),
        queryGrid, queryFeatures,
    )
    expected = (
        gp.evaluate(latent, queryGrid) + fixedEffect @ queryFeatures.T
        + 0.4
    )

    torch.testing.assert_close(prediction, expected)
