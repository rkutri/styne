"""
Gaussian Process Engine Comparison

Draw sample paths from direct, B-spline and DNA parametrisations of the same
Matérn Gaussian-process prior. Compare independent prior realisations with
evaluation through the stateful GaussianProcess interface.
"""

import numpy as np

from styne.gp import GaussianProcess
from styne.model.representation import BSpline1D
from styne.statistics import MaternCovariance1D, MaternCovariance2D
from styne.utility import UniformGrid

# check for matplotlib
try:
    import matplotlib.pyplot as plt
    hasMatplotlib = True
except ImportError:
    hasMatplotlib = False
    print("matplotlib not installed (pip install styne[plotting]); skipping plots.")

# fix seed
rng = np.random.default_rng(2026)


# --- SETUP ---

# covariance functions
lengthscale = 0.2
smoothness = 1.5
variance = 1.0

covariance1D = MaternCovariance1D(lengthscale, smoothness, variance)
covariance2D = MaternCovariance2D(0.5 * lengthscale, 0.5 * smoothness, variance)

# evaluation sites
resolution = 100
nSites = 300
sites = UniformGrid(0.0, 1.0, nSites)


# --- GP ENGINE DEFINITIONS ---

# direct engine: covariance matrix on a fixed grid
directGP = GaussianProcess.direct(
    UniformGrid(0.0, 1.0, resolution), covariance1D
)

# B-spline engine: coefficient vector in a spline expansion
bsplineExpansion = BSpline1D(resolution, degree=3, boundary=[0.0, 1.0])
bsplineExpansion.project(np.zeros(resolution))
bsplineGP = GaussianProcess.bspline(covariance1D, bsplineExpansion)

# DNA engine: Fourier-based white-noise parametrisation
dnaGP = GaussianProcess.dna(covariance1D, q=resolution, d=1)

engines = {
    "direct": directGP,
    "bspline": bsplineGP,
    "dna": dnaGP,
}

for name, gp in engines.items():
    gp.sites = sites
    print(f"{name:8s}: parameter dim = {gp.parameterDimension}")


# --- INDEPENDENT PRIOR REALISATIONS ---

nSamples = 2000
nPaths = 5
interior = (sites.axis >= 0.1) & (sites.axis <= 0.9)

print(f"\ndrawing {nSamples} independent prior realisations per engine ...")

samplerVar = {}
for name, gp in engines.items():

    samples = np.array([
        gp.sampler.generate_realisation(rng=rng).function.evaluate(sites)
        for _ in range(nSamples)
    ])

    samplerVar[name] = samples.var(axis=0)
    band = samplerVar[name][interior]

    print(f"  {name:8s}: marginal var on interior "
          f"[{band.min():.3f}, {band.max():.3f}]")


# --- STATELESS EVALUATION ---

print(f"\nevaluating {nSamples} parameter states per engine via gp.at_sites(z) ...")

atSitesVar = {}
for name, gp in engines.items():

    samples = np.empty((nSamples, nSites))

    for k in range(nSamples):
        realisation = gp.sampler.generate_realisation(rng=rng)
        samples[k] = gp.at_sites(realisation.coordinate)

    atSitesVar[name] = samples.var(axis=0)
    band = atSitesVar[name][interior]
    discrepancy = np.max(np.abs(atSitesVar[name][interior]
                                - samplerVar[name][interior]))

    print(f"  {name:8s}: marginal var on interior "
          f"[{band.min():.3f}, {band.max():.3f}], "
          f"max discrepancy={discrepancy:.3e}")


# --- POSTPROCESSING ---

if hasMatplotlib:

    fig, axes = plt.subplots(2, len(engines), figsize=(14, 7), sharey=True)

    for col, (name, gp) in enumerate(engines.items()):

        # independent prior realisations
        for _ in range(nPaths):
            path = gp.sampler.generate_realisation(
                rng=rng).function.evaluate(sites)
            axes[0, col].plot(sites.axis, path, lw=0.7, alpha=0.7)

        axes[0, col].plot(sites.axis, samplerVar[name],
                          "k--", lw=1.5, label="marginal var")
        axes[0, col].set_title(f"{name} — prior realisations")
        axes[0, col].legend(fontsize=8)

        # stateless evaluation of explicit GaussianProcess coefficients
        for _ in range(nPaths):
            realisation = gp.sampler.generate_realisation(rng=rng)
            axes[1, col].plot(
                sites.axis, gp.at_sites(realisation.coordinate),
                lw=0.7, alpha=0.7)

        axes[1, col].plot(sites.axis, atSitesVar[name],
                          "k--", lw=1.5, label="marginal var")
        axes[1, col].set_title(f"{name} — gp.at_sites(z)")
        axes[1, col].set_xlabel("x")
        axes[1, col].legend(fontsize=8)

    for ax in axes.flat:
        ax.set_ylim(-3.0, 3.0)

    axes[0, 0].set_ylabel("field value")
    axes[1, 0].set_ylabel("field value")

    fig.suptitle("Gaussian-process engines: prior draws and stateful evaluation")
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.25)
    plt.savefig("gp_sampling.png", dpi=150)
    plt.close()
    print("\nsaved gp_sampling.png")


# --- DNA COEFFICIENT PRIOR ---

specCov = dnaGP.measure.covariance
marginalVariances = specCov.marginalVariance
logDet = specCov.log_determinant()

zSample = dnaGP.measure.generate_realisation(rng=rng)
logPrior = dnaGP.measure.density.evaluate_log(zSample)

print("\nDNA coefficient prior:")
print(f"  marginal variances [{marginalVariances.min():.3e}, "
      f"{marginalVariances.max():.3e}]")
print(f"  log|C| = {logDet:.4f}")
print(f"  log-density of one prior draw = {logPrior:.4f}")


# --- 2D DNA REALISATION ---

resolution2D = 100
gp2D = GaussianProcess.dna(covariance2D, q=resolution2D, d=2)

realisation2D = gp2D.sampler.generate_realisation(rng=rng)
nativeField = realisation2D.function.evaluate_native()

nGrid = resolution2D + 2
field = nativeField.reshape(nGrid, nGrid)

print(f"\n2D DNA GP: parameter dim = {gp2D.parameterDimension}")
print(f"native grid: {nGrid} x {nGrid} ({nativeField.size} nodes)")

if hasMatplotlib:

    plt.figure(figsize=(5, 4))
    plt.imshow(
        field, origin="lower", extent=[0, 1, 0, 1],
        cmap="RdBu_r", vmin=-2.5, vmax=2.5,
    )
    plt.colorbar()
    plt.title(f"DNA GP sample — 2D Matérn, native {nGrid}x{nGrid} grid")
    plt.tight_layout()
    plt.savefig("dna_sample_2d.png", dpi=150)
    plt.close()
    print("saved dna_sample_2d.png")
