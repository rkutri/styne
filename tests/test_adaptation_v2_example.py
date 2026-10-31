from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from examples.runtime import as_numpy
from styne.backend import get_backend
from styne.mcmc import MetropolisedRandomWalk
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.method.mrw import MRWProposal
from styne.parameter import Vector
from styne.statistics import Gaussian, IIDCovarianceMatrix


example = import_module('examples.05_adaptation_v2')


@pytest.fixture(params=['numpy', 'jax', 'pytorch'])
def backend(request):
    if request.param != 'numpy':
        pytest.importorskip('torch' if request.param == 'pytorch' else request.param)
    return get_backend(request.param)


@pytest.fixture(scope='module', params=['numpy', 'jax', 'pytorch'])
def execution(request):
    if request.param != 'numpy':
        pytest.importorskip('torch' if request.param == 'pytorch' else request.param)
    path = Path(__file__).resolve().parents[1] / 'examples' / '05_adaptation_v2.py'
    output = StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        patch.setattr(sys, 'argv', [str(path), '--backend', request.param, '--smoke'])
        with redirect_stdout(output):
            namespace = runpy.run_path(str(path), run_name='__main__')
    return namespace, output.getvalue()


def test_binomial_data_have_three_strongly_correlated_covariates(execution):
    namespace, _ = execution
    features = namespace['features']
    successes = namespace['successes']
    assert features.shape == (400, 3)
    correlations = np.corrcoef(features, rowvar=False)
    assert np.all(correlations[np.triu_indices(3, k=1)] > 0.95)
    assert np.std(features[:, 1]) > 3 * np.std(features[:, 2])
    assert np.all((successes >= 0) & (successes <= namespace['nTrials']))
    assert np.any(successes > 1)
    assert namespace['nTrials'] == 10


def test_posterior_is_binomial_logit_plus_the_common_gaussian_prior(execution):
    namespace, _ = execution
    backend = namespace['backend']
    beta = np.array([0.3, -0.2, 0.4], dtype='float32')
    eta = namespace['features'].astype('float64') @ beta
    expected = np.sum(namespace['successes'] * eta - 10 * np.logaddexp(0, eta))
    expected -= 0.5 * np.sum(beta.astype('float64') ** 2) / 6.25
    actual = namespace['target'].evaluate_log(Vector(backend.asarray(beta)))
    np.testing.assert_allclose(as_numpy(actual), expected, rtol=2e-6, atol=1e-4)


@pytest.mark.parametrize('covariance', ['diagonal', 'full'])
def test_composed_adaptation_preserves_initial_iid_variance(backend, covariance):
    start = Vector(backend.zeros(3, dtype='float32'))
    template = MRWProposal(IIDCovarianceMatrix(3, backend.asarray(0.04)))
    rule = example.CovarianceScaleAdaptation(covariance)
    state = rule.initial_state(start, template)
    prepared = rule.prepare_proposal(template, state)
    factor = as_numpy(prepared.covariance.to_cholesky())
    np.testing.assert_allclose(factor @ factor.T, 0.04 * np.eye(3), atol=1e-7)


def test_full_covariance_scaling_does_not_refactor_between_scheduled_updates(monkeypatch):
    backend = get_backend('numpy')
    start = Vector(backend.zeros(3, dtype='float32'))
    target = Gaussian(IIDCovarianceMatrix(3, backend.asarray(1.0)), start)
    sampler = MetropolisedRandomWalk(
        target.density, IIDCovarianceMatrix(3, backend.asarray(0.01)),
        DummyDiagnostics(), rng=backend.random_state(12),
        adaptation=example.CovarianceScaleAdaptation('full', interval=10),
    )
    calls = []
    original = backend.cholesky

    def count_cholesky(matrix):
        calls.append(matrix.copy())
        return original(matrix)

    monkeypatch.setattr(backend, 'cholesky', count_cholesky)
    sampler.run(25, start)
    assert len(calls) == 2
    state = sampler.currentState.adaptationState
    assert state.scaleState.stepCount == 25
    assert state.shapeState.moments.nData == 25
    fixed = sampler.freeze_adaptation()
    fixed.run(20, sampler.lastState)
    assert len(calls) == 2


def test_three_methods_produce_production_ess_and_a_comparison_table(execution):
    namespace, output = execution
    results = namespace['results']
    assert len(results) == 3
    for result in results.values():
        assert result['ess'].shape == (3,)
        assert np.all(np.isfinite(result['ess']))
        assert np.all(result['ess'] > 0)
        assert 0 <= result['acceptance'] <= 1
        assert result['seconds'] > 0
        assert np.all(np.isfinite(result['posteriorMean']))
        assert np.all(np.linalg.eigvalsh(result['proposalCovariance']) > 0)
    isotropic, diagonal, full = [result['proposalCovariance']
                                for result in results.values()]
    np.testing.assert_allclose(isotropic, isotropic[0, 0] * np.eye(3), atol=1e-7)
    np.testing.assert_array_equal(diagonal, np.diag(np.diag(diagonal)))
    assert np.any(np.abs(full - np.diag(np.diag(full))) > 1e-6)
    assert all(name in output for name in results)
    assert all(column in output for column in ('ESS beta1', 'ESS beta2', 'ESS beta3',
                                               'Min ESS', 'Min ESS/s'))
