import numpy as np
import pytest

from styne.backend import get_backend
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.localised import (
    LocalisedSurrogateDensity, LocalisedSurrogateTransitionMeasure,
)
from styne.mcmc.method.dart import (
    DART, DARTFactory, GaussianDARTProposal, LocalisedSurrogateTransition,
)
from styne.mcmc.method.dartdirect import DirectDARTProposal
from styne.mcmc.method.mlda import MLDAFactory
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.mcmc.method.ratio import NormalisingRatioEstimator, RatioEstimator
from styne.mcmc.mixture import Mixture
from styne.parameter import Vector
from styne.statistics import DenseCovarianceMatrix, Gaussian, IIDCovarianceMatrix
from styne.statistics.covariance import SemidefiniteCovarianceMatrix


def make_factory(backendName='numpy'):
    backend = get_backend(backendName)
    dtype = 'float32' if backendName == 'jax' else 'float64'
    approximation = Gaussian(
        DenseCovarianceMatrix(backend.asarray([[2., 0.3], [0.3, 1.]], dtype=dtype)),
        Vector(backend.asarray([0.2, -0.1], dtype=dtype)),
    )
    factory = DARTFactory()
    factory.target = approximation.density
    factory.surrogate = [approximation.density]
    factory.approximation = [approximation]
    factory.penaltyMetric = [0.7]
    factory.rootLength = [4]
    factory.root.beta = 0.5
    return backend, factory


def test_factory_inherits_mlda_and_preserves_length_alias():
    _, factory = make_factory()
    assert isinstance(factory, MLDAFactory)
    assert factory.rootLength is factory.nChain
    factory.nChain = [7]
    assert factory.rootLength == [7]
    assert factory.level_burnin(0) == 1
    for length in range(12):
        assert NormalisingRatioEstimator.default_burnin(length) == length // 4
    assert isinstance(factory.create(), DART)


def test_legacy_regularisation_preserves_penalty_and_getter_meanings():
    _, factory = make_factory()
    weights = np.array([0.5, 2.0])
    factory.regularisation = [0.7]
    factory.spectralWeights = weights
    sampler = factory.create()
    localised = sampler.proposal.density.with_location(Vector(np.zeros(2)))
    point = Vector(np.array([0.8, -0.3]))
    assert factory.regularisation == [0.7]
    assert localised.regularisation == 0.7
    assert localised.spectralWeights is weights
    assert localised.log_localisation(point) == pytest.approx(
        -0.5 * 0.7 * np.sum((weights * point.coordinate) ** 2)
    )
    factory.spectralWeights = None
    factory.penaltyMetric = [DenseCovarianceMatrix(np.eye(2))]
    with pytest.raises(ValueError, match='no scalar'):
        _ = factory.regularisation


def test_mixture_reuses_geometry_and_clears_single_density_metadata(monkeypatch):
    backend, factory = make_factory()
    factory.create()
    assert factory.localisedDensity is not None
    factory.components = [{'beta': 0.2}, {'beta': 0.8}]
    factory.weights = [0.3, 0.7]
    cholesky = backend.cholesky
    calls = []

    def counted_cholesky(matrix):
        calls.append(1)
        return cholesky(matrix)

    monkeypatch.setattr(backend, 'cholesky', counted_cholesky)
    mixture = factory.create()
    first, second = mixture.components
    assert isinstance(mixture, Mixture)
    assert first.target is second.target is factory.target
    assert first.proposal.density is second.proposal.density
    assert first.proposal.density.reference.covariance is (
        second.proposal.density.reference.covariance
    )
    assert first.proposal.measure.mcmc.proposal.beta == 0.2
    assert second.proposal.measure.mcmc.proposal.beta == 0.8
    assert factory.localisedDensity is None
    assert len(calls) == 1


def test_penalty_reference_reuses_validated_dense_metric_factor(monkeypatch):
    backend, factory = make_factory()
    metric = DenseCovarianceMatrix(np.array([[2., 0.3], [0.3, 1.]]), scaling=1.7)

    def unexpected_factorisation(matrix):
        raise AssertionError('Validated penalty metric was factorised again.')

    monkeypatch.setattr(backend, 'cholesky', unexpected_factorisation)
    density = LocalisedSurrogateDensity(
        penaltyMetric=metric, surrogateDensity=factory.target, penaltyReference=True
    ).with_location(Vector(np.array([0.4, -0.2])))
    expected = np.linalg.inv(metric.apply(np.eye(2)))
    np.testing.assert_allclose(density.reference.covariance.apply(np.eye(2)), expected)


def test_explicit_metric_default_supports_semidefinite_penalty_without_reference():
    _, factory = make_factory()
    metric = SemidefiniteCovarianceMatrix(np.diag([0.0, 2.0]))
    density = LocalisedSurrogateDensity(
        penaltyMetric=metric, surrogateDensity=factory.target
    )
    assert density.reference is None
    assert not density.penaltyReference
    assert density.log_localisation(Vector(np.array([1.0, 2.0]))) == -4.0
    legacy = LocalisedSurrogateDensity(0.7, 1.0, factory.target)
    assert legacy.penaltyReference
    assert isinstance(legacy.reference, Gaussian)


def test_mixture_validates_the_retained_count_for_each_component():
    _, factory = make_factory()
    factory.components = [{'rootLength': 4}, {'rootLength': 1}]
    with pytest.raises(ValueError, match='retains 1 samples'):
        factory.create()


def test_reused_legacy_estimator_uses_selected_components_metric():
    _, factory = make_factory()
    density = factory.create().proposal.density

    def measure(metric):
        localised = LocalisedSurrogateDensity(
            penaltyMetric=metric, surrogateDensity=density.surrogateDensity,
            approximation=density.approximation,
        )
        root = MetropolisedRandomWalk(
            localised, IIDCovarianceMatrix(2, 0.2), DummyDiagnostics()
        )
        return LocalisedSurrogateTransitionMeasure(root, 4)

    selected = measure(3.0)
    estimator = RatioEstimator(measure(0.2), 0, 1, 'gaussian')
    reused = LocalisedSurrogateTransition(selected, 0, 1, estimator)
    fresh = LocalisedSurrogateTransition(selected, 0, 1)
    start = Vector(np.array([0.5, -0.7]))
    actual, _ = reused.propose(start, np.random.default_rng(31))
    expected, _ = fresh.propose(start, np.random.default_rng(31))
    np.testing.assert_array_equal(
        actual.proposed.parameter.coordinate, expected.proposed.parameter.coordinate
    )
    assert actual.auxiliary['logCorrection'] == pytest.approx(
        expected.auxiliary['logCorrection']
    )
    assert not hasattr(estimator, '_surrogateMeasure')


@pytest.mark.parametrize('method', ['importance', 'gaussian', 'bridge'])
def test_ratio_estimator_keeps_batch_axes_and_can_reuse_configuration(method):
    estimator = NormalisingRatioEstimator(type=method)
    state = Vector(np.array([[0.0, 0.1], [0.4, 0.2]]))
    proposal = Vector(np.array([[0.7, 0.3], [0.2, -0.5]]))
    trajectory = np.random.default_rng(45).normal(size=(6, 2, 2))
    reverse = np.random.default_rng(46).normal(size=(6, 2, 2))
    for metric in [0.5, DenseCovarianceMatrix(np.array([[2., 0.4], [0.4, 1.]]))]:
        together = estimator.log_ratio_estimate(
            state, proposal, trajectory, reverse, penaltyMetric=metric
        )
        separately = [estimator.log_ratio_estimate(
            Vector(state.coordinate[index]), Vector(proposal.coordinate[index]),
            trajectory[:, index], reverse[:, index], penaltyMetric=metric,
        ) for index in range(2)]
        np.testing.assert_allclose(together, separately)


def test_gaussian_dart_correction_matches_exact_proposal_density():
    _, factory = make_factory()
    approximation = factory.approximation[0]
    metric = DenseCovarianceMatrix(np.array([[0.7, 0.2], [0.2, 0.9]]))
    proposal = GaussianDARTProposal(approximation, metric)
    current = Vector(np.array([0.8, -0.5]))
    record, _ = proposal.propose(current, np.random.default_rng(11))
    proposed = record.proposed.parameter
    precision = np.linalg.inv(approximation.covariance.to_dense())
    penalty = metric.to_dense()
    covariance = np.linalg.inv(precision + penalty)

    def local_law(point):
        mean = covariance @ (precision @ approximation.mean.coordinate
                             + penalty @ point.coordinate)
        return Gaussian(DenseCovarianceMatrix(covariance), Vector(mean)).density

    expected = (local_law(proposed).evaluate_log(current)
                - local_law(current).evaluate_log(proposed))
    assert record.auxiliary['logCorrection'] == pytest.approx(expected)


def test_direct_dart_rejects_inconsistent_legacy_covariance():
    _, factory = make_factory()
    with pytest.raises(ValueError, match='inconsistent'):
        DirectDARTProposal(0.5, 0.1, factory.approximation[0],
                           DenseCovarianceMatrix(np.eye(2)))


def test_direct_dart_preserves_consistent_scaled_legacy_covariance():
    _, factory = make_factory()
    approximation = factory.approximation[0]
    precision = 0.5 * approximation.covariance.apply_inverse(np.eye(2)) + 0.1 * np.eye(2)
    covariance = DenseCovarianceMatrix(np.linalg.inv(precision) / 2.0, scaling=2.0)
    proposal = DirectDARTProposal(0.5, 0.1, approximation, covariance)
    assert proposal._localised.reference.covariance is covariance


def test_factory_mixture_compiles_with_independent_batched_jax_choices():
    pytest.importorskip('jax')
    backend, factory = make_factory('jax')
    factory.components = [{'beta': 0.2}, {'beta': 0.8}]
    sampler = factory.create()
    start = Vector(backend.zeros((4, 2), dtype='float32'))
    result = sampler.compiled_trajectory(12)(start, backend.random_state(37))
    assert result.coordinates.shape == (12, 4, 2)
    selected = np.asarray(result.sums['selected'])
    np.testing.assert_array_equal(selected.sum(axis=1), np.full(4, 12))
    assert np.unique(selected[:, 0]).size > 1
    assert np.all(np.isfinite(result.coordinates))
