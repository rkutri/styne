"""
Ratio-estimator stability vs total surrogate evaluation cost.
Comparison of IS and Cumulant estimators (MALA sub-chain).
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'examples'))
from styne.statistics.mixture import GaussianMixtureDensity
from scipy.special import logsumexp

# pyrefly: ignore [missing-import]
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
nRuns = 10_000

accGoal = 0.55
nTuning = 1000
seed = 2026

tempering = 1.0

mixWeights = (0.7, 0.3)
a = 0.5
sigma = 1.0

# Production localisation grid. Spans the stable regime (gamma = 0.05,
# stable at every d) up to the IS-breakdown regime (gamma = 2.5). The
# admissible threshold gamma/L ~ d^{-1/2} is crossed panel by panel as d
# grows, so the breakdown marches leftward across the figure. Values
# beyond ~2.5 are outside any usable regime (both estimators degenerate,
# and the tuner fails), so they are excluded deliberately.
gammas = [0.05, 0.25, 1.0, 2.5]
plotJitter = 1.04

rng = np.random.default_rng(seed)
resultData = {}


CACHE = Path(__file__).parent / 'joblib_caches' / f"{Path(__file__).stem}.joblib"
if hasJoblib:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
# Recompute when PARAMS change; reuse the cache otherwise. Set True to
# force a fresh production run regardless of the cache.
FORCE = True

PARAMS = dict(
    thinning=thinning,
    burnInFraction=burnInFraction,
    dims=dims,
    nBase=nBase,
    nMultipliers=nMultipliers,
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

from scipy.integrate import quad
_d_saved = None
def _check_truth():
    gam = 1.3
    x0, z0 = 0.4, -0.2
    dens = lambda y, c: (
        mixWeights[0] * np.exp(-0.5 * (y - a) ** 2 / sigma**2)
        + mixWeights[1] * np.exp(-0.5 * (y + a) ** 2 / sigma**2)
    ) * np.exp(-0.5 * gam * (y - c) ** 2)
    Nx = quad(dens, -30, 30, args=(x0,))[0]
    Nz = quad(dens, -30, 30, args=(z0,))[0]

    def log_truth_1d(point, gammaValue):
        s2 = sigma**2 + 1.0 / gammaValue
        terms = [
            np.log(wk)
            - 0.5 * 1 * np.log(2.0 * np.pi * s2)
            - 0.5 * ((point - mk) ** 2) / s2
            for wk, mk in zip(mixWeights, (a, -a))
        ]
        return logsumexp(terms)

    closedForm = log_truth_1d(z0, gam) - log_truth_1d(x0, gam)
    assert abs(closedForm - np.log(Nz / Nx)) < 1e-8, "truth formula broken"
_check_truth()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def compute_statistics(errors):
    """Median absolute relative error with interquartile range."""
    absErr = np.abs(errors)
    center = np.median(absErr, axis=0)
    lower = np.percentile(absErr, 25, axis=0)
    upper = np.percentile(absErr, 75, axis=0)
    yErr = np.vstack([center - lower, upper - center])
    return center, yErr


# -------------------------------------------------------------------------
# Experiment loop
# -------------------------------------------------------------------------
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
    if not hasJoblib:
        print("Warning: joblib is not installed. Caching is disabled.")
    print(f"{'Dim':<4} | {'Gamma':<10} | {'Status'}")
    print("-" * 32)

    for d in dims:
        surrCov = IIDCovarianceMatrix(d, sigma**2)
        mean1 = a * np.ones(d)
        mean2 = -a * np.ones(d)
        g1 = GaussianDensity(surrCov)
        g1.mean = Vector(mean1)
        g2 = GaussianDensity(surrCov)
        g2.mean = Vector(mean2)
        surrogateDensity = GaussianMixtureDensity([g1, g2], weights=mixWeights)

        x = Vector(a * np.ones(d))

        nValues = [nBase[d] * m for m in nMultipliers]

        for gammaValue in gammas:
            print(
                f"{d:<4} | {gammaValue:<10.2e} | Computing...",
                end="\r",
            )

            dartDensity = LocalisedSurrogateDensity(
                gammaValue, tempering, surrogateDensity
            )
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

            errIs = np.empty((nRuns, len(nValues)))
            errCu = np.empty_like(errIs)

            def log_truth(point, gammaValue):
                s2 = sigma**2 + 1.0 / gammaValue
                terms = [
                    np.log(wk)
                    - 0.5 * d * np.log(2.0 * np.pi * s2)
                    - 0.5 * np.sum((point - mk) ** 2) / s2
                    for wk, mk in zip(mixWeights, (mean1, mean2))
                ]
                return logsumexp(terms)

            for ni, nValue in enumerate(nValues):
                surrogateMeasure = DARTMeasure(
                    rootMCMC,
                    nValue,
                )
                surrogateMeasure.location = x

                burnIn = int(burnInFraction * nValue)
                isEstimator = RatioEstimator(
                    surrogateMeasure, burnIn, thinning, type="is"
                )
                cumulantEstimator = RatioEstimator(
                    surrogateMeasure, burnIn, thinning, type="cumulant"
                )

                for k in range(nRuns):
                    z = surrogateMeasure.generate_realisation(rng=rng)

                    logIs = isEstimator.log_ratio_estimate(x, z)
                    logCu = cumulantEstimator.log_ratio_estimate(x, z)

                    logTruth = log_truth(z.coordinate, gammaValue) - log_truth(x.coordinate, gammaValue)

                    errIs[k, ni] = np.abs(logIs - logTruth)
                    errCu[k, ni] = np.abs(logCu - logTruth)

                    if gammaValue == gammas[0] and ni == len(nValues) - 1:
                        traj = surrogateMeasure.chain.trajectory
                        samplesX = np.array(traj)
                        proj = samplesX @ (np.ones(d) / np.sqrt(d))
                        if not (np.any(proj > 0) and np.any(proj < 0)):
                            print(f"WARNING: chain did not cross components (d={d}, gamma={gammaValue})")

            centerIs, errorIs = compute_statistics(errIs)
            centerCu, errorCu = compute_statistics(errCu)

            resultData[(d, gammaValue)] = (
                centerIs,
                errorIs,
                centerCu,
                errorCu,
                nValues,
            )

            print(
                f"{d:<4} | {gammaValue:<10.2e} | Done         "
            )

    if hasJoblib:
        data = {'params': PARAMS, 'results': resultData}
        joblib.dump(data, CACHE, compress=3)

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------
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
    fig, axes = plt.subplots(1, len(gammas), figsize=(figWidth, 4.5))

    for idx, gammaValue in enumerate(gammas):
        ax = axes[idx]
        style_axes(ax)
        ax.set_title(rf"$\gamma = {gammaValue}$", fontweight="bold")

        for dIdx, d in enumerate(dims):
            stats = resultData[(d, gammaValue)]
            centerIs, errorIs, centerCu, errorCu, nValues = stats

            # Base evaluation cost for the dimension
            xBaseVal = nBase[d]
            xVals = np.array(nMultipliers) * xBaseVal

            cmap = plt.get_cmap("plasma")
            color = cmap(dIdx / (len(dims) - 1) * 0.8)

            xCu = xVals * plotJitter
            yCu = centerCu
            errCu = errorCu

            # Cumulant (solid)
            ax.errorbar(
                xCu,
                yCu,
                yerr=errCu,
                fmt="-o",
                color=color,
                markersize=8.0,
                capsize=2.0,
                capthick=2.,
                elinewidth=2.,
                zorder=4,
            )

            xIs = xVals / plotJitter
            yIs = centerIs
            errIs = errorIs

            # Importance Sampling (dashed, increased alpha)
            ax.errorbar(
                xIs,
                yIs,
                yerr=errIs,
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

        # Enable minor ticks for logarithmic scale
        subDecades = (2, 3, 4, 5, 6, 7, 8, 9)
        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=subDecades, numticks=20)
        )
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=subDecades, numticks=20)
        )

        # Grid line styling (both major and minor grids for log scale)
        ax.grid(
            True,
            which="major",
            axis="both",
            color="0.75",
            linestyle="-",
            linewidth=0.5,
            zorder=0,
        )
        ax.grid(
            True,
            which="minor",
            axis="both",
            color="0.88",
            linestyle=":",
            linewidth=0.3,
            zorder=0,
        )

        ax.set_xlabel(r"$n$")

        if idx == 0:
            ax.set_ylabel("abs. error")

    # Legended patches & lines setup
    legHandles = []
    cmap = plt.get_cmap("plasma")
    for dIdx, d in enumerate(dims):
        color = cmap(dIdx / (len(dims) - 1) * 0.8)
        patch = mpatches.Patch(
            color=color,
            label=f"d = {d}"
        )
        legHandles.append(patch)

    # Labels match the manuscript symbols: R_G (solid) and R_IS (dashed).
    legHandles.append(
        mlines.Line2D(
            [], [],
            color="0.3",
            linestyle="-",
            marker="o",
            markersize=8.0,
            label=r"$\hat{R}_{\mathrm{G}}$",
        )
    )
    legHandles.append(
        mlines.Line2D(
            [], [],
            color="0.3",
            linestyle="--",
            marker="x",
            markersize=8.0,
            alpha=0.75,
            label=r"$\hat{R}_{\mathrm{IS}}$",
        )
    )

    # Legend centering: if number of subplots is odd, center on the middle axes.
    # If even, center on the boundary of the two middle axes.
    midIdx = len(gammas) // 2
    if len(gammas) % 2 == 1:
        legAx = axes[midIdx]
        bboxAnchor = (0.5, -0.18)
    else:
        legAx = axes[midIdx - 1]
        bboxAnchor = (1.15, -0.18)

    legAx.legend(
        handles=legHandles,
        loc="upper center",
        bbox_to_anchor=bboxAnchor,
        ncol=6,
        handlelength=1.6,
        columnspacing=1.2,
        borderpad=0.4,
        labelspacing=0.35,
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.1,
        top=0.9,
        wspace=0.18,
    )

    figuresDir = Path(__file__).parent.parent / "figures"
    figuresDir.mkdir(parents=True, exist_ok=True)
    figPath = figuresDir / "manuscript_stability.pdf"

    fig.savefig(
        figPath,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    print(f"\nSaved: {figPath}")
    plt.close(fig)
else:
    print("Warning: matplotlib is not installed. Skipping plotting.")
