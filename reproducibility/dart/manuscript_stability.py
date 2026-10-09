"""
Ratio-estimator stability vs root chain length.
Comparison of IS and Cumulant estimators (MALA root MCMC).
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from styne.statistics.mixture import GaussianMixtureDensity
from scipy.special import logsumexp
from scipy.integrate import quad

from manuscript_boilerplate import (
    hasMatplotlib, hasJoblib, plt, joblib
)
if hasMatplotlib:
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker

from styne.parameter.vector import Vector
from styne.statistics.gaussian import Gaussian, GaussianDensity
from styne.statistics.covariance import DiagonalCovarianceMatrix, IIDCovarianceMatrix
from styne.mcmc.localised import (
    LocalisedSurrogateDensity,
    LocalisedSurrogateTransitionMeasure as DARTMeasure,
)
from styne.mcmc.method.mala import MALAFactory
from styne.mcmc.method.ratio import RatioEstimator
from styne.utility.tuning import MALATuner, LangevinTunerConfig

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
thinning = 1
burnInFraction = 0.3

dims = [4, 8, 16, 32]
nBase = {d: int(10 * np.sqrt(d / 2)) for d in dims}
nMultipliers = [1, 2, 4, 8, 16, 32, 64, 128]
nLow = 9
nRuns = 20_000

accGoal = 0.55
nTuning = 5000
seed = 2026

tempering = 1.0

mixWeights = (0.7, 0.3)
a = 0.5
sigma = 1.0

# Parallelism. 16 cells (4 dims x 4 gammas), one worker holds one cell.
nJobs = 4

gammas = [0.05, 0.25, 1.0, 2.5]
plotJitter = 1.04

CACHE = Path(__file__).parent / 'joblib_caches' / f"{Path(__file__).stem}.joblib"
FORCE = True

PARAMS = dict(
    thinning=thinning,
    burnInFraction=burnInFraction,
    dims=dims,
    nBase=nBase,
    nMultipliers=nMultipliers,
    nLow=nLow,
    nRuns=nRuns,
    accGoal=accGoal,
    nTuning=nTuning,
    seed=seed,
    tempering=tempering,
    mixWeights=mixWeights,
    a=a,
    sigma=sigma,
    gammas=gammas,
)


def _check_truth():
    """Validate the 1D closed-form log-ratio against quadrature."""
    gamma = 1.3
    x0, z0 = 0.4, -0.2
    density = lambda y, c: (
        mixWeights[0] * np.exp(-0.5 * (y - a) ** 2 / sigma**2)
        + mixWeights[1] * np.exp(-0.5 * (y + a) ** 2 / sigma**2)
    ) * np.exp(-0.5 * gamma * (y - c) ** 2)
    normX = quad(density, -30, 30, args=(x0,))[0]
    normZ = quad(density, -30, 30, args=(z0,))[0]

    def log_truth_1d(point, gammaValue):
        sigmaSquared = sigma**2 + 1.0 / gammaValue
        terms = [
            np.log(wk)
            - 0.5 * 1 * np.log(2.0 * np.pi * sigmaSquared)
            - 0.5 * ((point - mk) ** 2) / sigmaSquared
            for wk, mk in zip(mixWeights, (a, -a))
        ]
        return logsumexp(terms)

    closedForm = log_truth_1d(z0, gamma) - log_truth_1d(x0, gamma)
    assert abs(closedForm - np.log(normZ / normX)) < 1e-8, "truth formula broken"


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def compute_statistics(errors):
    """Median absolute value of relative error with interquartile range."""

    absoluteError = np.abs(errors)

    center = np.median(absoluteError, axis=0)
    lower = np.percentile(absoluteError, 25, axis=0)
    upper = np.percentile(absoluteError, 75, axis=0)

    yError = np.vstack([center - lower, upper - center])
    return center, yError


def run_cell(d, gammaValue, childSeed):
    """
        run one (d, gamma) cell. Self-contained for parallel dispatch.

    """

    rng = np.random.default_rng(childSeed)
    warnings = []

    surrogateCovariance = IIDCovarianceMatrix(d, sigma**2)
    mean1 = a * np.ones(d)
    mean2 = -a * np.ones(d)

    g1 = GaussianDensity(surrogateCovariance, Vector(mean1))
    g2 = GaussianDensity(surrogateCovariance, Vector(mean2))

    surrogateDensity = GaussianMixtureDensity([g1, g2], weights=mixWeights)

    x = Vector(a * np.ones(d))
    nValues = [nLow] + [nBase[d] * m for m in nMultipliers]

    dartDensity = LocalisedSurrogateDensity(gammaValue, tempering, surrogateDensity)
    dartDensity.location = x

    mcmcFactory = MALAFactory()
    mcmcFactory.target = dartDensity
    mcmcFactory.rng = rng

    try:
        config = LangevinTunerConfig(accGoal, nTuning)
        rootMCMC = MALATuner(mcmcFactory, x, config).tune()

    except RuntimeError:
        mcmcFactory.stepSize = 1e-5
        rootMCMC = mcmcFactory.create()

    errorIs = np.empty((nRuns, len(nValues)))
    errorCumulant = np.empty_like(errorIs)

    def log_truth(point, gammaValue):

        sigmaSquared = sigma**2 + 1.0 / gammaValue
        terms = [
            np.log(wk)
            - 0.5 * d * np.log(2.0 * np.pi * sigmaSquared)
            - 0.5 * np.sum((point - mk) ** 2) / sigmaSquared
            for wk, mk in zip(mixWeights, (mean1, mean2))
        ]
        return logsumexp(terms)

    for ni, nValue in enumerate(nValues):

        surrogateMeasure = DARTMeasure(rootMCMC, nValue)
        surrogateMeasure.location = x

        burnIn = int(burnInFraction * nValue)
        isEstimator = RatioEstimator(surrogateMeasure, burnIn, thinning, type="is")
        cumulantEstimator = RatioEstimator(
            surrogateMeasure, burnIn, thinning, type="cumulant"
        )

        for k in range(nRuns):
            z = surrogateMeasure.generate_realisation(rng=rng)

            logIs = isEstimator.log_ratio_estimate(x, z)
            logCu = cumulantEstimator.log_ratio_estimate(x, z)

            logTruth = log_truth(z.coordinate, gammaValue) - log_truth(
                x.coordinate, gammaValue
            )

            errorIs[k, ni] = np.abs(logIs - logTruth)
            errorCumulant[k, ni] = np.abs(logCu - logTruth)

            if gammaValue == gammas[0] and ni == len(nValues) - 1:

                trajectory = surrogateMeasure.chain.trajectory
                samplesX = np.array(trajectory)

                projection = samplesX @ (np.ones(d) / np.sqrt(d))

                if not (np.any(projection > 0) and np.any(projection < 0)):
                    warnings.append(
                        f"chain did not cross components (d={d}, gamma={gammaValue})"
                    )

    centerIs, errorIs = compute_statistics(errorIs)
    centerCumulant, errorCumulant = compute_statistics(errorCumulant)

    return (
        d,
        gammaValue,
        centerIs,
        errorIs,
        centerCumulant,
        errorCumulant,
        nValues,
        warnings,
    )


# =========================================================================
# Driver
# =========================================================================
if __name__ == "__main__":

    _check_truth()

    if hasJoblib:
        CACHE.parent.mkdir(parents=True, exist_ok=True)

    resultData = {}
    cacheLoaded = False

    if hasJoblib and CACHE.exists() and not FORCE:

        try:
            data = joblib.load(CACHE)

            if data.get('params') == PARAMS:
                resultData = data['results']
                print("Loaded results from cache.")
                cacheLoaded = True

            else:
                print("Stale cache detected, recomputing...")

        except Exception:
            print("Failed to load cache, recomputing...")

    if not cacheLoaded:

        cells = [(d, g) for d in dims for g in gammas]
        childSeeds = np.random.SeedSequence(seed).spawn(len(cells))

        if hasJoblib:
            from joblib import Parallel, delayed

            print(f"Dispatching {len(cells)} cells on {nJobs} workers...")
            results = Parallel(n_jobs=nJobs, backend="loky", verbose=10)(
                delayed(run_cell)(d, g, s)
                for (d, g), s in zip(cells, childSeeds)
            )
        else:
            print("Warning: joblib not installed. Running serially, no cache.")
            results = [run_cell(d, g, s) for (d, g), s in zip(cells, childSeeds)]

        for (
            d,
            g,
            centerIs,
            errorIs,
            centerCumulant,
            errorCumulant,
            nValues,
            cellWarnings,
        ) in results:
            resultData[(d, g)] = (
                centerIs,
                errorIs,
                centerCumulant,
                errorCumulant,
                nValues,
            )
            for warningMsg in cellWarnings:
                print(f"WARNING: {warningMsg}")

        if hasJoblib:
            joblib.dump({'params': PARAMS, 'results': resultData}, CACHE, compress=3)

    # ---------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------
    if hasMatplotlib:

        plt.close('all')
        plt.style.use(Path(__file__).parent / 'manuscript.mplstyle')

        def style_axes(axis):

            axis.tick_params(direction="out", length=3.0, width=0.8)

            for side in ["top", "right"]:
                axis.spines[side].set_visible(False)
            for side in ["bottom", "left"]:
                axis.spines[side].set_linewidth(0.8)

        figWidth = 4.5 * len(gammas) + 0.5
        fig, axes = plt.subplots(1, len(gammas), figsize=(figWidth, 4.5), sharey=True)

        for idx, gammaValue in enumerate(gammas):

            ax = axes[idx]
            style_axes(ax)
            ax.set_title(rf"$\gamma = {gammaValue}$", fontweight="bold")

            for dIdx, d in enumerate(dims):

                (
                    centerIs,
                    errorIs,
                    centerCumulant,
                    errorCumulant,
                    nValues,
                ) = resultData[(d, gammaValue)]

                xValues = np.asarray(nValues, dtype=float)

                cmap = plt.get_cmap("plasma")
                color = cmap(dIdx / (len(dims) - 1) * 0.8)

                ax.errorbar(
                    xValues * plotJitter,
                    centerCumulant,
                    yerr=errorCumulant,
                    fmt="-o",
                    color=color,
                    markersize=8.0,
                    capsize=2.0,
                    capthick=2.,
                    elinewidth=2.,
                    zorder=4,
                )

                ax.errorbar(
                    xValues / plotJitter,
                    centerIs,
                    yerr=errorIs,
                    fmt="--x",
                    color=color,
                    markersize=8.0,
                    alpha=0.75,
                    capsize=1.5,
                    capthick=1.875,
                    elinewidth=1.875,
                    zorder=3,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_box_aspect(1)

            subDecades = (2, 3, 4, 5, 6, 7, 8, 9)
            ax.xaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=subDecades, numticks=20)
            )
            ax.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=subDecades, numticks=20)
            )
            ax.tick_params(axis="y", which="both", left=True, labelleft=True)

            ax.grid(
                True, which="major", axis="both",
                color="0.75", linestyle="-", linewidth=0.5, zorder=0,
            )
            ax.grid(
                True, which="minor", axis="both",
                color="0.88", linestyle=":", linewidth=0.3, zorder=0,
            )

            ax.set_xlabel(r"$n$")
            if idx == 0:
                ax.set_ylabel("absolute error")

        legendHandles = []
        cmap = plt.get_cmap("plasma")
        for dIdx, d in enumerate(dims):
            color = cmap(dIdx / (len(dims) - 1) * 0.8)
            legendHandles.append(mpatches.Patch(color=color, label=f"d = {d}"))

        legendHandles.append(
            mlines.Line2D(
                [], [], color="0.3", linestyle="-", marker="o",
                markersize=8.0, label=r"$\hat{R}_{\mathrm{G}}$",
            )
        )
        legendHandles.append(
            mlines.Line2D(
                [], [], color="0.3", linestyle="--", marker="x",
                markersize=8.0, alpha=0.75, label=r"$\hat{R}_{\mathrm{IS}}$",
            )
        )

        midIdx = len(gammas) // 2
        if len(gammas) % 2 == 1:
            legendAxis = axes[midIdx]
            bboxAnchor = (0.5, -0.18)
        else:
            legendAxis = axes[midIdx - 1]
            bboxAnchor = (1.15, -0.18)

        legendAxis.legend(
            handles=legendHandles,
            loc="upper center",
            bbox_to_anchor=bboxAnchor,
            ncol=6,
            handlelength=1.6,
            columnspacing=1.2,
            borderpad=0.4,
            labelspacing=0.35,
        )

        fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.18,
        )

        figuresDir = Path(__file__).parent.parent / "figures"
        figuresDir.mkdir(parents=True, exist_ok=True)
        figPath = figuresDir / "manuscript_stability.pdf"

        fig.savefig(figPath, bbox_inches="tight", pad_inches=0.02)
        print(f"\nSaved: {figPath}")
        plt.close(fig)
    else:
        print("Warning: matplotlib is not installed. Skipping plotting.")
