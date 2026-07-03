"""Sample a Matern Gaussian process through three engines (direct Cholesky,
B-spline expansion, DNA Fourier) and compare their sample paths and empirical
marginal standard deviation."""

import numpy as np

from styne.gp import GaussianProcess
from styne.statistics import MaternCovariance1D
from styne.model.representation import BSpline1D
from styne.utility import UniformGrid

try:
    import matplotlib.pyplot as plt
    hasMatplotlib = True
except ImportError:
    hasMatplotlib = False
    print("matplotlib not installed (pip install styne[plotting]); skipping plot.")

rng = np.random.default_rng(1)
covariance = MaternCovariance1D(0.3, 1.5, 1.0)
sites = UniformGrid(0.0, 1.0, 300)
resolution = 100
nSamples = 2000

# BSpline1D needs its knots set via project() before use in an engine.
# Fewer basis functions keep the induced prior well-conditioned; finer
# bases inflate the marginal variance near the boundary and eventually
# lose positive definiteness.
bspline = BSpline1D(25)
bspline.project(np.zeros(25))

engines = {
    "direct": GaussianProcess.direct(UniformGrid(0.0, 1.0, resolution), covariance),
    "bspline": GaussianProcess.bspline(covariance, bspline),
    "dna": GaussianProcess.dna(covariance, resolution, d=1),
}

marginalStd = {}
for name, gp in engines.items():
    sampler = gp.sampler
    samples = np.array(
        [sampler.draw(rng).function.evaluate(sites) for _ in range(nSamples)]
    )
    marginalStd[name] = samples.std(axis=0)
    interior = (sites.axis >= 0.1) & (sites.axis <= 0.9)
    band = marginalStd[name][interior]
    print(f"{name:8s}: interior marginal std [{band.min():.3f}, {band.max():.3f}]")

if hasMatplotlib:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (name, gp) in zip(axes, engines.items()):
        sampler = gp.sampler
        for _ in range(5):
            ax.plot(sites.axis, sampler.draw(rng).function.evaluate(sites),
                    lw=0.7, alpha=0.7)
        ax.plot(sites.axis, marginalStd[name], "k--", lw=1.5, label="marginal std")
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.legend()
    axes[0].set_ylabel("field value")
    fig.suptitle("DNA holds the marginal variance to the boundary; "
                 "direct and B-spline degrade at the edges")
    plt.tight_layout()
    plt.savefig("gp_sampling.png", dpi=150)
    print("saved gp_sampling.png")
