"""
Tests for the Barker and Standard acceptance strategies.

Covers:
  1. Unit tests for both strategies.
  2. Integration smoke test: MRW on a 2D standard Gaussian.
  3. Detailed balance verification: log alpha(l) - log alpha(-l) == l.
"""

import numpy as np
import pytest

from styne.mcmc.acceptance import StandardAcceptance, BarkerAcceptance
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.mcmc.diagnostics import AcceptanceRateDiagnostics
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.interface import DensityInterface
from styne.parameter.vector import Vector


# ---------------------------------------------------------------------------
# Minimal 2D standard Gaussian target for the integration test
# ---------------------------------------------------------------------------

class StandardGaussian2D(DensityInterface):

    @property
    def domainType(self):
        return Vector

    @property
    def domainDimension(self):
        return 2

    def evaluate_log(self, param):
        x = param.coordinate
        return float(-0.5 * x @ x)


# ---------------------------------------------------------------------------
# 1. Unit tests
# ---------------------------------------------------------------------------

class TestStandardAcceptance:

    def setup_method(self):
        self.acc = StandardAcceptance()

    def test_zero(self):
        assert self.acc.log_probability(0.0) == pytest.approx(0.0)

    def test_negative(self):
        assert self.acc.log_probability(-5.0) == pytest.approx(-5.0)

    def test_positive(self):
        assert self.acc.log_probability(5.0) == pytest.approx(0.0)

    def test_nan(self):
        assert self.acc.log_probability(float('nan')) == float('-inf')

    def test_neg_inf(self):
        assert self.acc.log_probability(float('-inf')) == float('-inf')

    def test_pos_inf(self):
        assert self.acc.log_probability(float('inf')) == pytest.approx(0.0)


class TestBarkerAcceptance:

    def setup_method(self):
        self.acc = BarkerAcceptance()

    def test_zero(self):
        # 0 - logaddexp(0, 0) = 0 - log(2) = -log(2)
        assert self.acc.log_probability(0.0) == pytest.approx(-np.log(2.0))

    def test_negative(self):
        l = -5.0
        expected = l - float(np.logaddexp(0., l))
        assert self.acc.log_probability(l) == pytest.approx(expected)

    def test_positive(self):
        l = 5.0
        expected = l - float(np.logaddexp(0., l))
        assert self.acc.log_probability(l) == pytest.approx(expected)

    def test_nan(self):
        assert self.acc.log_probability(float('nan')) == float('-inf')

    def test_neg_inf(self):
        assert self.acc.log_probability(float('-inf')) == float('-inf')

    def test_pos_inf(self):
        # l - logaddexp(0, l) → l - l = 0 as l → +inf
        assert self.acc.log_probability(float('inf')) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Integration test: MRW on 2D standard Gaussian
# ---------------------------------------------------------------------------

def run_mrw(acceptance, n_steps=5000, seed=42):
    np.random.seed(seed)
    target = StandardGaussian2D()
    propCov = IIDCovarianceMatrix(2, 0.5)
    diagnostics = AcceptanceRateDiagnostics()
    chain = MetropolisedRandomWalk(target, propCov, diagnostics,
                                   acceptance=acceptance)

    init = Vector(np.zeros(2))
    chain.run(n_steps, init)

    samples = np.array(chain.chain.trajectory)
    rate = chain.diagnostics.global_acceptance_rate()
    return samples, rate


class TestIntegration:

    def test_standard_acceptance_rate(self):
        _, rate = run_mrw(StandardAcceptance())
        assert rate > 0.0

    def test_barker_acceptance_rate(self):
        _, rate = run_mrw(BarkerAcceptance())
        assert rate > 0.0

    def test_standard_mean(self):
        samples, _ = run_mrw(StandardAcceptance())
        assert np.abs(samples.mean(axis=0)).max() < 0.5

    def test_barker_mean(self):
        samples, _ = run_mrw(BarkerAcceptance())
        assert np.abs(samples.mean(axis=0)).max() < 0.5

    def test_post_hoc_replacement(self):
        """Verify that swapping acceptance post-construction works."""
        np.random.seed(0)
        target = StandardGaussian2D()
        propCov = IIDCovarianceMatrix(2, 0.5)
        diagnostics = AcceptanceRateDiagnostics()
        chain = MetropolisedRandomWalk(target, propCov, diagnostics)

        chain.acceptance = BarkerAcceptance()
        assert isinstance(chain.acceptance, BarkerAcceptance)

        chain.run(500, Vector(np.zeros(2)))
        assert chain.diagnostics.global_acceptance_rate() > 0.0


# ---------------------------------------------------------------------------
# 3. Detailed balance: log alpha(l) - log alpha(-l) == l
# ---------------------------------------------------------------------------

class TestDetailedBalance:

    @pytest.mark.parametrize("l", [-10., -5., -1., 0., 1., 5., 10.])
    def test_standard(self, l):
        acc = StandardAcceptance()
        diff = acc.log_probability(l) - acc.log_probability(-l)
        assert diff == pytest.approx(l, abs=1e-12)

    @pytest.mark.parametrize("l", [-10., -5., -1., 0., 1., 5., 10.])
    def test_barker(self, l):
        acc = BarkerAcceptance()
        diff = acc.log_probability(l) - acc.log_probability(-l)
        assert diff == pytest.approx(l, abs=1e-10)
