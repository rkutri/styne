import pytest

import numpy as np

from numpy.random import default_rng
from numpy.linalg import norm
from scipy.stats import gaussian_kde

from styne.parameter.vector import Vector
from styne.mcmc.method.mrw import MRWFactory
from styne.statistics.covariance import IIDCovarianceMatrix
from tests.testSetup import (
    covariance_matrix,
    GaussianTargetDensity,
)
from styne.utility.postprocessing import integrated_autocorrelation



relativeTolerance = 0.25

# After IAT thinning, two estimated marginal standard errors approximate a 95%
# marginal CLT scale. With fixed seeds, this is a deterministic regression margin,
# not a per-run flake probability.
meanStandardErrors = 2.0


@pytest.mark.parametrize("dim", [1, 4])
@pytest.mark.parametrize("kappa", [1., 5.])
def test_mrw_moments(dim, kappa):

    tgtMean = [Vector(np.zeros(dim)),
               Vector(15. * np.ones(dim))]
    tgtCov = [covariance_matrix(dim, kappa), 0.5 *
              covariance_matrix(dim, kappa)]

    # set up Gaussian target densities
    target = [GaussianTargetDensity(m, c) for m, c in zip(tgtMean, tgtCov)]

    # set up metropolised random walk
    rng = default_rng(1111)
    builder = MRWFactory()
    builder.rng = rng

    hRef = 8.
    builder.proposalCovariance = IIDCovarianceMatrix(
        dim, (hRef / dim)**2 / kappa)
    builder.target = target[0]

    mcmc = builder.create()

    # run configuration
    nSteps = dim * int(kappa) * int(3000)
    initState = Vector(np.ones(dim) * (-1.))

    for i in range(len(target)):

        # test switching target densities
        mcmc.target = target[i]

        # run chain
        mcmc.run(nSteps, initState)
        states = np.array(mcmc.chain.trajectory)

        # postprocess
        burnin = 1000

        thinningStep = int(np.ceil(integrated_autocorrelation(states[burnin:], 'max')))
        samples = states[burnin::thinningStep]

        # check pointwise error of estimated mean
        meanEst = np.mean(samples, axis=0)
        sampleStd = np.std(samples, axis=0, ddof=1)
        meanSE = sampleStd / np.sqrt(len(samples))
        meanError = np.abs(meanEst - tgtMean[i].coordinate)
        assert np.all(meanError < meanStandardErrors * meanSE)

        # check frobenius error of estimated covariance
        covEst = np.cov(samples, rowvar=False)
        relCovError = np.linalg.norm(covEst - tgtCov[i], ord='fro') \
            / np.linalg.norm(tgtCov[i], ord='fro')

        assert relCovError < relativeTolerance


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("kappa", [1, 5])
def test_mrw_density(dim, kappa):

    # setup target density
    tgtMean = Vector(np.zeros(dim))
    tgtCov = covariance_matrix(dim, kappa)
    target = GaussianTargetDensity(tgtMean, tgtCov)

    rng = default_rng(1112)

    # set up metropolised random walk
    builder = MRWFactory()
    builder.rng = rng

    hRef = 9.
    builder.proposalCovariance = IIDCovarianceMatrix(dim, hRef / kappa)
    builder.target = target

    mcmc = builder.create()

    # run chain
    nSteps = dim * int(kappa) * int(3000)
    initState = Vector(np.ones(dim) * (-1.))
    mcmc.run(nSteps, initState)

    states = np.array(mcmc.chain.trajectory)

    # postprocess
    burnin = 1000
    thinningStep = int(np.ceil(integrated_autocorrelation(states[burnin:], 'max')))
    samples = states[burnin::thinningStep]

    # construct density estimate from samples
    nGridPerDim = 50
    axes = [np.linspace(-3., 3., nGridPerDim) for _ in range(dim)]
    mesh = np.meshgrid(*axes, indexing='ij')
    mesh = np.vstack([m.ravel() for m in mesh])
    gridSpacing = axes[0][1] - axes[0][0]
    volumeElement = gridSpacing ** dim

    # Kernel Density Estimation
    kde = gaussian_kde(samples.T)
    densityEst = kde(mesh)

    # evaluate actual target density on mesh
    targetDensity = np.exp(
        np.array([
            target.evaluate_log(Vector(coord))
            for coord in mesh.T
        ])
    )

    # normalize both densities
    densityEst /= np.sum(densityEst) * volumeElement
    targetDensity /= np.sum(targetDensity) * volumeElement

    # compute relative error
    error = norm(densityEst - targetDensity) / norm(targetDensity)
    assert error < relativeTolerance


def test_progress_preserves_trajectory(capsys):
    trajectories = []
    for progress in (False, True):
        builder = MRWFactory()
        builder.target = GaussianTargetDensity(Vector(np.zeros(2)), np.eye(2))
        builder.proposalCovariance = IIDCovarianceMatrix(2, 0.5)
        builder.rng = default_rng(1112)
        sampler = builder.create()
        sampler.run(10, Vector(np.zeros(2)), progress=progress)

        assert bool(capsys.readouterr().err) is progress
        trajectories.append(np.array(sampler.chain.trajectory))

    np.testing.assert_array_equal(trajectories[0], trajectories[1])
