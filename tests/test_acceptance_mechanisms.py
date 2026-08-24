"""
Tests that BarkerAcceptance and StandardAcceptance target the same invariant measure.
"""

import numpy as np
import pytest

from styne.mcmc.acceptance import BarkerAcceptance, StandardAcceptance


class Gaussian1D:

    def __init__(self, mean, var):
        self._mean = mean
        self._var = var

    def evaluate_log(self, param):
        x = float(np.atleast_1d(param.coordinate)[0])
        return -0.5 * (x - self._mean) ** 2 / self._var


def run_mrw_chain(acceptance_cls, n_steps=30_000, seed=42):
    from numpy.random import default_rng
    from styne.mcmc.method.mrw import MetropolisedRandomWalk
    from styne.mcmc.diagnostics import DummyDiagnostics
    from styne.statistics.covariance import IIDCovarianceMatrix
    from styne.parameter.scalar import Scalar

    rng = default_rng(seed)

    targetMean = 2.0
    targetVar = 1.5
    tgt = Gaussian1D(targetMean, targetVar)

    propCov = IIDCovarianceMatrix(1, 0.5)
    chain = MetropolisedRandomWalk(tgt, propCov, DummyDiagnostics(),
                                   acceptance=acceptance_cls(), rng=rng)

    init = Scalar(0.0)
    chain.run(n_steps, init)

    samples = np.array(chain.chain.trajectory)
    burnin = 2000
    return samples[burnin:], targetMean, targetVar


class TestBarkerInvariantMeasure:
    """
    Barker acceptance satisfies detailed balance and targets the same invariant
    distribution as standard MH.  Verified by checking that empirical moments
    of a long chain match the known Gaussian moments.
    """

    MEAN_TOL = 0.15
    VAR_TOL = 0.20

    @pytest.mark.parametrize("acceptance_cls",
                             [StandardAcceptance, BarkerAcceptance])
    def test_mean_matches_target(self, acceptance_cls):
        samples, targetMean, _ = run_mrw_chain(acceptance_cls)
        assert abs(np.mean(samples) - targetMean) < self.MEAN_TOL, (
            f"{acceptance_cls.__name__}: empirical mean {np.mean(samples):.3f} "
            f"deviates from target {targetMean} by more than {self.MEAN_TOL}"
        )

    @pytest.mark.parametrize("acceptance_cls",
                             [StandardAcceptance, BarkerAcceptance])
    def test_variance_matches_target(self, acceptance_cls):
        samples, _, targetVar = run_mrw_chain(acceptance_cls)
        assert abs(np.var(samples) - targetVar) < self.VAR_TOL, (
            f"{acceptance_cls.__name__}: empirical var {np.var(samples):.3f} "
            f"deviates from target {targetVar} by more than {self.VAR_TOL}"
        )

    def test_barker_and_standard_agree(self):
        stdSamples, _, _ = run_mrw_chain(StandardAcceptance, seed=0)
        barSamples, _, _ = run_mrw_chain(BarkerAcceptance, seed=0)

        assert abs(np.mean(stdSamples) - np.mean(barSamples)) < 0.15
        assert abs(np.var(stdSamples) - np.var(barSamples)) < 0.25
