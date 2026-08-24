"""
1D Linear Regression in a Bayesian Framework

Recover slope and intercept from noisy data
with a Metropolised-Random-Walk sampler.
"""
import numpy as np

from styne.model import LinearForwardMap
from styne.parameter import Vector
from styne.statistics import (
    Data, Gaussian, GaussianResponse, RegressionLikelihood,
    UnnormalisedPosterior, DiagonalCovarianceMatrix, IIDCovarianceMatrix,
)
from styne.mcmc import MRWFactory
from styne.utility import integrated_autocorrelation

# check for matplotlib
try:
    import matplotlib.pyplot as plt
    hasMatplotlib = True
except ImportError:
    hasMatplotlib = False
    print("matplotlib not installed (pip install styne[plotting]); skipping plot.")

# fix seed
rng = np.random.default_rng(2026)


# --- SETUP ---

# ground truth
noiseVar = 0.16
trueParam = [-0.7, 2.3]  # [intercept, slope]
trueIntercept, trueSlope = trueParam

# measurement locations
nObs = 60
locations = np.linspace(0.0, 1.0, nObs)
data = Data(dimension=nObs, design=locations[:, None])

# columns: [intercept, slope]
features = np.column_stack((np.ones(nObs), locations))

# synthetic data generation
measurement = features @ trueParam + np.sqrt(noiseVar) * rng.standard_normal(nObs)
data.measurement = measurement


# --- PROBLEM FORMULATION ---

# prior definition
priorCov = IIDCovarianceMatrix(2, 6.0)
prior = Gaussian(priorCov, mean=Vector(np.zeros(2)))

# likelihood definition
noiseModel = GaussianResponse(IIDCovarianceMatrix(nObs, noiseVar))
forwardModel = LinearForwardMap(features)
likelihood = RegressionLikelihood(data, forwardModel, noiseModel)

# posterior definition
posterior = UnnormalisedPosterior(prior, likelihood)


# --- INFERENCE ---

factory = MRWFactory()
factory.target = posterior
factory.proposalCovariance = DiagonalCovarianceMatrix([0.02, 0.08])
factory.rng = rng

sampler = factory.create()

# run mcmc
nSteps = 20000
initState = Vector(np.zeros(2))
sampler.run(nSteps, initState)

# discard burn-in
nBurnIn = 5000
trajectory = np.array(sampler.chain.trajectory)[nBurnIn:]
intercept, slope = trajectory.mean(axis=0)

# postprocessing
iat = integrated_autocorrelation(trajectory, method="max")
acceptanceRate = sampler.diagnostics.global_acceptance_rate()
interceptVar, slopeVar = trajectory.var(axis=0)

print(f"true:      intercept={trueIntercept:.3f}, slope={trueSlope:.3f}")
print(f"posterior: intercept={intercept:.3f} (var={interceptVar:.4f}), "
      f"slope={slope:.3f} (var={slopeVar:.4f})")
print(f"acceptance rate={acceptanceRate:.3f}, iat={iat:.1f}")

if hasMatplotlib:
    # posterior predictive band: each draw is a line, take pointwise quantiles
    lines = trajectory[:, 0][:, None] + trajectory[:, 1][:, None] * locations[None, :]
    lower, upper = np.percentile(lines, [2.5, 97.5], axis=0)

    plt.figure(figsize=(7, 4))
    plt.scatter(locations, measurement, s=20, label="data")
    plt.plot(locations, slope * locations + intercept, "r-", label="posterior mean")
    plt.fill_between(locations, lower, upper, color="r", alpha=0.2,
                     label="95% credible band")
    plt.legend()
    plt.savefig("quickstart_fit.png", dpi=150)
    print("saved quickstart_fit.png")
