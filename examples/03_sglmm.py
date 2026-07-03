"""Spatial GLMM with a latent Matern field and Poisson counts. Recover the
latent intensity field from count observations at scattered sites, with the
covariance hyperparameters held fixed."""

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

try:
    import matplotlib.pyplot as plt
    hasMatplotlib = True
except ImportError:
    hasMatplotlib = False
    print("matplotlib not installed (pip install styne[plotting]); skipping plot.")

rng = np.random.default_rng(0)

covariance = MaternCovariance2D(0.3, 1.5, 1.0)
gp = GaussianProcess.dna(covariance, q=8, d=2)
latentDim = gp.parameter.dimension

# DNA parametrises the field in whitened coordinates, so the prior on the
# latent block is standard normal.
nObs = 200
meanLevel = 1.5
obsSites = Grid(rng.uniform(0.0, 1.0, (nObs, 2)))
model = SGLMM(gp, obsSites)

zTrue = rng.standard_normal(latentDim)
model.interpolate(Vector(zTrue))
model.evaluate()
counts = rng.poisson(np.exp(model.evaluation + meanLevel))

data = Data(1, obsSites.to_array())
data.measurement = counts[:, None]
likelihood = SGLMMLikelihood(data, model, PoissonResponse())

prior = Gaussian(IIDCovarianceMatrix(latentDim, 1.0), Vector(np.zeros(latentDim)))
posterior = UnnormalisedPosterior(prior, likelihood)

factory = MALAFactory()
factory.target = posterior
factory.stepSize = 0.08
factory.rng = rng
sampler = factory.create()

print("running MALA (a few seconds) ...")
sampler.run(5000, Vector(np.zeros(latentDim)))
trajectory = np.array(sampler.chain.trajectory)[1500:]
zMean = trajectory.mean(axis=0)

dense = UniformGrid((0.0, 1.0, 40), (0.0, 1.0, 40))
gp.sites = dense
gp.parameter.coordinate = zTrue
fieldTrue = gp.at_sites()
gp.parameter.coordinate = zMean
fieldRecovered = gp.at_sites()

correlation = np.corrcoef(fieldTrue, fieldRecovered)[0, 1]
print(f"posterior mean vs truth: field correlation = {correlation:.3f}")

if hasMatplotlib:
    grid = fieldTrue.reshape(40, 40)
    recovered = fieldRecovered.reshape(40, 40)
    vmin, vmax = grid.min(), grid.max()
    fig, (axTrue, axRec) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, title in [(axTrue, grid, "true field"),
                             (axRec, recovered, "posterior mean")]:
        im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                       vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_title(title)
    axTrue.scatter(obsSites.to_array()[:, 0], obsSites.to_array()[:, 1],
                   s=6, c="k", alpha=0.4)
    fig.colorbar(im, ax=(axTrue, axRec), shrink=0.8)
    plt.savefig("sglmm_field.png", dpi=150)
    print("saved sglmm_field.png")
