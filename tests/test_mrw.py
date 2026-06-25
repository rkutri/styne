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



RTOL = 0.25
ATOL = 0.2


@pytest.mark.slow
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
        assert np.allclose(
            meanEst, tgtMean[i].coordinate, atol=ATOL, rtol=RTOL)

        # check frobenius error of estimated covariance
        covEst = np.cov(samples, rowvar=False)
        relCovError = np.linalg.norm(covEst - tgtCov[i], ord='fro') \
            / np.linalg.norm(tgtCov[i], ord='fro')

        assert relCovError < RTOL


@pytest.mark.slow
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("kappa", [1, 5])
@pytest.mark.parametrize("progress", [False, True])
def test_mrw_density(dim, kappa, progress, capsys):

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
    mcmc.run(nSteps, initState, progress=progress)
    
    captured = capsys.readouterr()
    if progress:
        assert len(captured.err) > 0, "Expected progress bar output in stderr"
    else:
        assert len(captured.err) == 0, "Expected no progress bar output when progress=False"

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
    densityEst /= np.sum(densityEst) * (10. / nGridPerDim)**dim
    targetDensity /= np.sum(targetDensity) * (10. / nGridPerDim)**dim

    # compute relative error
    error = norm(densityEst - targetDensity) / norm(targetDensity)
    assert error < RTOL
