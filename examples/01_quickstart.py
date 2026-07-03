"""Bayesian 1D linear regression: recover slope and intercept from noisy data
with a random-walk Metropolis sampler."""

import numpy as np

from styne.model import LinearModel
from styne.parameter import Vector
from styne.statistics import (
    Data, Gaussian, GaussianResponse, RegressionLikelihood,
    UnnormalisedPosterior, DiagonalCovarianceMatrix, IIDCovarianceMatrix,
)
from styne.mcmc import MRWFactory

try:
    import matplotlib.pyplot as plt
    hasMatplotlib = True
except ImportError:
    hasMatplotlib = False
    print("matplotlib not installed (pip install styne[plotting]); skipping plot.")

rng = np.random.default_rng(0)

nObs = 60
trueSlope, trueIntercept, noiseStd = 2.3, -0.7, 0.4

xData = np.linspace(0.0, 1.0, nObs)
features = np.column_stack((xData, np.ones(nObs)))
measurement = features @ np.array([trueSlope, trueIntercept]) \
    + noiseStd * rng.standard_normal(nObs)

data = Data(dimension=nObs, design=features)
data.measurement = measurement

likelihood = RegressionLikelihood(
    data, LinearModel(features),
    GaussianResponse(IIDCovarianceMatrix(nObs, noiseStd**2))
)
prior = Gaussian(DiagonalCovarianceMatrix(np.full(2, 10.0)), Vector(np.zeros(2)))
posterior = UnnormalisedPosterior(prior, likelihood)

factory = MRWFactory()
factory.target = posterior
factory.proposalCovariance = DiagonalCovarianceMatrix(np.full(2, 0.01))
factory.rng = rng
sampler = factory.create()

sampler.run(20000, Vector(np.zeros(2)))
trajectory = np.array(sampler.chain.trajectory)[5000:]
slope, intercept = trajectory.mean(axis=0)

print(f"true:      slope={trueSlope:.3f}, intercept={trueIntercept:.3f}")
print(f"posterior: slope={slope:.3f}, intercept={intercept:.3f}")

if hasMatplotlib:
    plt.figure(figsize=(7, 4))
    plt.scatter(xData, measurement, s=20, label="data")
    plt.plot(xData, slope * xData + intercept, "r-", label="posterior mean")
    plt.legend()
    plt.savefig("quickstart_fit.png", dpi=150)
    print("saved quickstart_fit.png")
