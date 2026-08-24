"""
Spatial GLMM in a Bayesian Framework

Recover a latent Gaussian field from Poisson observations
with a Metropolis-adjusted Langevin sampler.
"""

import numpy as np

from styne.gp import GaussianProcess
from styne.model import SGLMM
from styne.parameter import Vector
from styne.statistics import (
    MaternCovariance2D, Data, PoissonResponse, SGLMMLikelihood,
    Gaussian, IIDCovarianceMatrix, UnnormalisedPosterior,
)
from styne.mcmc import MALAFactory
from styne.utility import Grid, UniformGrid

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

# latent Gaussian field
spatialDim = 2
truthResolution = 32
resolution = 8

covariance = MaternCovariance2D(0.3, 1.5, 1.0)
truthGP = GaussianProcess.dna(covariance, q=truthResolution, d=spatialDim)
gp = GaussianProcess.dna(covariance, q=resolution, d=spatialDim)
latentDim = gp.parameter.dimension

# DNA uses whitened latent coefficients, so the prior is standard normal
# on the coordinate vector.
zTrue = truthGP.sampler.generate_realisation(rng=rng)

# measurement locations
nObs = 200
obsSites = Grid(rng.uniform(0.0, 1.0, (nObs, spatialDim)))
model = SGLMM(gp, obsSites)

# synthetic data generation
counts = rng.poisson(np.exp(truthGP.evaluate(zTrue.coordinate, obsSites)))

data = Data(dimension=1, design=obsSites.to_array())
data.measurement = counts[:, None]


# --- PROBLEM FORMULATION ---

# prior definition
priorCov = IIDCovarianceMatrix(latentDim, 1.0)
prior = Gaussian(priorCov, mean=Vector(np.zeros(latentDim)))

# likelihood definition
likelihood = SGLMMLikelihood(data, model, PoissonResponse())

# posterior definition
posterior = UnnormalisedPosterior(prior, likelihood)


# --- INFERENCE ---

factory = MALAFactory()
factory.target = posterior
factory.stepSize = 0.08
factory.gradient = posterior.evaluate_log_gradient
factory.rng = rng

sampler = factory.create()

# run mcmc
nSteps = 10_000
initState = Vector(np.zeros(latentDim))
sampler.run(nSteps, initState, progress=True, description="Sampling MALA")

# discard burn-in
nBurnIn = 1500
trajectory = np.array(sampler.chain.trajectory)[nBurnIn:]
zMean = trajectory.mean(axis=0)

# diagnostics
acceptanceRate = sampler.diagnostics.global_acceptance_rate()
print(f"acceptance rate={acceptanceRate:.3f}")


# --- POSTPROCESSING ---

# compare posterior mean to the true latent field on a dense grid
gridRes = 40
dense = UniformGrid((0.0, 1.0, gridRes), (0.0, 1.0, gridRes))

fieldTrue = truthGP.evaluate(zTrue.coordinate, dense)

fieldRecovered = gp.evaluate(zMean, dense)

correlation = np.corrcoef(fieldTrue, fieldRecovered)[0, 1]
print(f"posterior mean vs truth: field correlation = {correlation:.3f}")

if hasMatplotlib:
    grid = fieldTrue.reshape(gridRes, gridRes)
    recovered = fieldRecovered.reshape(gridRes, gridRes)

    vmin = min(grid.min(), recovered.min())
    vmax = max(grid.max(), recovered.max())

    fig, (axTrue, axRec) = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained"
    )
    for ax, field, title in [
        (axTrue, grid, "true latent field"),
        (axRec, recovered, "posterior mean"),
    ]:
        im = ax.imshow(
            field, origin="lower", extent=[0, 1, 0, 1],
            vmin=vmin, vmax=vmax, cmap="RdBu_r",
        )
        ax.set_title(title)

    axTrue.scatter(
        obsSites.to_array()[:, 0], obsSites.to_array()[:, 1],
        s=6, c="k", alpha=0.4,
    )

    fig.colorbar(im, ax=(axTrue, axRec), shrink=0.8)
    plt.savefig("sglmm_field.png", dpi=150)
    plt.close()
    print("saved sglmm_field.png")
