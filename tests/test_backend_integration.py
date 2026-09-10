import numpy as np
import pytest

from styne.backend import BackendUnavailableError, get_backend, infer_backend
from styne.gp.gaussianprocess import GaussianProcess
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.mcmc.method.pcn import PreconditionedCrankNicolson
from styne.model.forwardmap import ForwardMap
from styne.model.representation.bspline import BSpline1D
from styne.model.sglmm import SGLMM
from styne.parameter import BlockParameter, Vector
from styne.statistics.bayes import UnnormalisedPosterior
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.data import Data
from styne.statistics.gaussian import Gaussian, GaussianDensity
from styne.statistics.radonnikodym import RadonNikodym
from styne.statistics.likelihood import RegressionLikelihood
from styne.statistics.hierarchical import SGLMMLatentConditional
from styne.statistics.response import GaussianResponse
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import UniformGrid


@pytest.fixture(params=('numpy', 'pytorch', 'jax'))
def backend(request):
    try:
        return get_backend(request.param)
    except BackendUnavailableError:
        pytest.skip(f'{request.param} is not installed')


class GPEvaluationForwardMap(ForwardMap):

    def __init__(self, process, sites):
        self._evaluation = process.bind(sites)
        self._dimension = process.parameterDimension

    @property
    def pType(self):
        return Vector

    @property
    def pDim(self):
        return self._dimension

    def _prepare(self, parameter):
        return parameter.coordinate

    def _evaluate(self, preparedState):
        return self._evaluation.evaluate(preparedState)


def covariance(backend):
    return MaternCovariance1D(
        backend.asarray(0.4, dtype='float32'),
        1.5,
        backend.asarray(1.0, dtype='float32'),
    )


def process(representation, backend):
    if representation == 'direct':
        return GaussianProcess.direct(
            UniformGrid(0.0, 1.0, 5), covariance(backend)
        )
    if representation == 'bspline':
        return GaussianProcess.bspline(
            covariance(backend), BSpline1D(5, degree=3, boundary=[0.0, 1.0])
        )
    return GaussianProcess.dna(covariance(backend), q=2, d=1)


@pytest.mark.parametrize('representation', ('direct', 'bspline', 'dna'))
def test_gp_posterior_mrw_integration_preserves_backend(
        backend, representation):
    gp = process(representation, backend)
    sites = UniformGrid(0.1, 0.9, 3)
    data = Data(1, sites.to_array())
    data.measurement = backend.zeros((3, 1), dtype='float32')
    likelihood = RegressionLikelihood(
        data,
        GPEvaluationForwardMap(gp, sites),
        GaussianResponse(IIDCovarianceMatrix(
            3, backend.asarray(0.5, dtype='float32')
        )),
    )
    posterior = UnnormalisedPosterior(gp.measure, likelihood)
    sampler = MetropolisedRandomWalk(
        posterior,
        IIDCovarianceMatrix(
            gp.parameterDimension, backend.asarray(0.1, dtype='float32')
        ),
        DummyDiagnostics(),
    )
    current = sampler.initial_state(
        Vector(backend.zeros(gp.parameterDimension, dtype='float32'))
    )

    nextState, transition, _ = sampler.step(current, backend.random_state(5))

    assert infer_backend(current.logDensity) is backend
    assert infer_backend(nextState.parameter.coordinate) is backend
    assert nextState.parameter.coordinate.shape == (gp.parameterDimension,)
    assert transition.logAcceptanceProbability.shape == ()


def test_sglmm_posterior_mrw_integration_preserves_backend(backend):
    sites = UniformGrid(0.1, 0.9, 3)
    features = backend.asarray(
        [[1.0, -0.5], [0.2, 0.3], [-0.4, 0.7]], dtype='float32'
    )
    gp = GaussianProcess.dna(covariance(backend), q=2, d=1)
    model = SGLMM(gp, sites, features=features)
    latent = Vector(backend.zeros(gp.parameterDimension, dtype='float32'))
    fixed = Vector(backend.zeros(2, dtype='float32'))
    parameter = BlockParameter([latent, fixed])
    data = Data(1, sites.to_array())
    data.measurement = backend.zeros((3, 1), dtype='float32')
    likelihood = RegressionLikelihood(
        data,
        model,
        GaussianResponse(IIDCovarianceMatrix(
            3, backend.asarray(0.5, dtype='float32')
        )),
    )
    prior = Gaussian(
        IIDCovarianceMatrix(
            parameter.dimension, backend.asarray(1.0, dtype='float32')
        ),
        parameter,
    )
    sampler = MetropolisedRandomWalk(
        UnnormalisedPosterior(prior, likelihood),
        IIDCovarianceMatrix(
            parameter.dimension, backend.asarray(0.1, dtype='float32')
        ),
        DummyDiagnostics(),
    )
    current = sampler.initial_state(parameter)

    nextState, transition, _ = sampler.step(current, backend.random_state(5))

    assert infer_backend(current.logDensity) is backend
    assert infer_backend(nextState.parameter.coordinate) is backend
    assert isinstance(nextState.parameter, BlockParameter)
    assert nextState.parameter.coordinate.shape == (parameter.dimension,)
    assert transition.logAcceptanceProbability.shape == ()


def test_pcn_retarget_preserves_backend_and_uses_new_reference(backend):
    dtype = 'float32'
    oldReference = Gaussian(
        IIDCovarianceMatrix(1, backend.asarray(1.0, dtype=dtype)),
        Vector(backend.zeros(1, dtype=dtype)),
    )
    newReference = Gaussian(
        IIDCovarianceMatrix(1, backend.asarray(4.0, dtype=dtype)),
        Vector(backend.full((1,), 10.0, dtype=dtype)),
    )
    derivative = GaussianDensity(
        IIDCovarianceMatrix(1, backend.asarray(1.0, dtype=dtype)),
        Vector(backend.full((1,), 10.0, dtype=dtype)),
    )
    sampler = PreconditionedCrankNicolson(
        RadonNikodym(oldReference, derivative), 1.0, DummyDiagnostics()
    )
    sampler.target = RadonNikodym(newReference, derivative)
    expected, _ = newReference.sample(backend.random_state(7))
    current = sampler.initial_state(
        Vector(backend.full((1,), 10.0, dtype=dtype))
    )

    _, transition, _ = sampler.step(current, backend.random_state(7))

    assert infer_backend(transition.proposed.parameter.coordinate) is backend
    np.testing.assert_allclose(
        transition.proposed.parameter.coordinate, expected.coordinate
    )


def test_spatial_conditioning_preserves_backend_composition(backend):
    dtype = 'float32'
    sites = UniformGrid(0.1, 0.9, 3)
    gp = process('dna', backend)
    data = Data(1, sites.to_array())
    data.measurement = backend.zeros((3, 1), dtype=dtype)
    likelihood = RegressionLikelihood(
        data,
        SGLMM(gp, sites),
        GaussianResponse(IIDCovarianceMatrix(
            3, backend.asarray(0.5, dtype=dtype)
        )),
    )
    target = UnnormalisedPosterior(gp.measure, likelihood)
    conditional = SGLMMLatentConditional(target, gp)
    latent = Vector(backend.zeros(gp.parameterDimension, dtype=dtype))
    hyperparameters = Vector(backend.asarray(
        [-0.7, 0.2], dtype=dtype
    ))

    conditioned = conditional.condition(
        BlockParameter([latent, hyperparameters])
    )
    value = conditioned.evaluate_log(latent)

    assert infer_backend(value) is backend
    assert conditioned.reference is conditioned.gp.measure
    assert conditioned.derivative.model._gp is conditioned.gp
    assert conditional.gp is gp
