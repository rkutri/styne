"""
Trace plot comparison: MRW vs MALA vs 2-level DART vs 3-level DART
for two 1D bimodal (Gaussian mixture) targets.
"""

import numpy as np
from pathlib import Path

from manuscript_boilerplate import (
    hasMatplotlib, hasJoblib, plt, joblib
)
if hasMatplotlib:
    from matplotlib import rcParams

import styne.utility.postprocessing as ac

from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import GaussianDensity
from styne.parameter.scalar import Scalar
from styne.mcmc.localised import LocalisedSurrogateDensity
from styne.mcmc.method.dart import DARTFactory
from styne.mcmc.method.mrw import MRWFactory
from styne.mcmc.method.mala import MALAFactory
from styne.mcmc.method.pcn import PCNFactory
from styne.utility.tuning import (
    MRWTuner,
    MALATuner,
    PCNTuner,
    RWTunerConfig,
    LangevinTunerConfig,
)

from styne.statistics.mixture import GaussianMixtureDensity
from manuscript_style import METHOD_COLORS


# =========================================================================
# Global settings
# =========================================================================
randomSeed = 2026
np.random.seed(randomSeed)
DIM = 1
nTuning = 1000
nSteps = 55_000
burnin = 5000
plotSteps = 5000
initState = Scalar(0.0)

methodSpecs = [
    ("MRW", "mrw"),
    ("MALA", "mala"),
    ("2-level DART", "dart2"),
    ("3-level DART", "dart3"),
]

accGoals = {
    "mrw": 0.3,
    "mala": 0.5
}

temperingInfo = {
    "MRW": None,
    "MALA": None,
    "2-level DART": [0.5],
    "3-level DART": [0.25, 0.5],
}

nChainInfo = {
    "2-level DART": [20],
    "3-level DART": [20, 10],
}

colours = METHOD_COLORS

experiments = [
    {
        "separation": 4.,
        "modeVar": 0.4,
        "gamma": 0.05,
    },
    {
        "separation": 8.,
        "modeVar": 0.2,
        "gamma": 0.01,
    },
]


# =========================================================================
# Helpers
# =========================================================================
def build_dart(
        tgtDensity, factory, gammas, thetas, nChain, init, nTuning, accGoal):
    surr0 = LocalisedSurrogateDensity(
        regularisation=gammas[0],
        tempering=thetas[0],
        surrogateDensity=tgtDensity)
    surr0.location = init.clone()

    rootFactory = PCNFactory()
    rootFactory.target = surr0

    try:
        PCNTuner(rootFactory, init, RWTunerConfig(accGoal, nTuning)).tune()

    except RuntimeError:

        fallBackBeta = 0.25
        print(
            "Warning: Failed to tune root surrogate density. "
            f"Defaulting to {fallBackBeta:.2f}."
        )

        rootFactory.beta = fallBackBeta

    factory.regularisation = gammas
    factory.tempering = thetas
    factory.nChain = nChain
    factory.root = rootFactory
    return factory.create()


def make_target(separation, modeVar):
    modeCov = IIDCovarianceMatrix(DIM, modeVar)

    gauss1 = GaussianDensity(modeCov, Scalar(-0.5 * separation))
    gauss2 = GaussianDensity(modeCov, Scalar(0.5 * separation))

    return GaussianMixtureDensity([gauss1, gauss2])


def count_crossings(trace):
    """
    Number of mode transitions in a 1D trace.
    """
    signs = np.sign(trace)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum(np.abs(np.diff(signs)) > 0))


def run_experiment(separation, modeVar, gamma):
    tgtDensity = make_target(separation, modeVar)

    gammaInfo = {
        "2-level DART": [gamma],
        "3-level DART": [gamma, gamma],
    }

    init = Scalar(0.0)

    mrwFactory = MRWFactory()
    mrwFactory.target = tgtDensity
    mrw = MRWTuner(
        mrwFactory, init, RWTunerConfig(accGoals["mrw"], nTuning)
    ).tune()

    malaFactory = MALAFactory()
    malaFactory.target = tgtDensity
    mala = MALATuner(
        malaFactory, init, LangevinTunerConfig(accGoals["mala"], nTuning)
    ).tune()

    dartFactory2 = DARTFactory("pcn")
    dartFactory2.target = tgtDensity
    dartFactory2.surrogate = [tgtDensity]
    dart2 = build_dart(
        tgtDensity, dartFactory2,
        gammaInfo["2-level DART"], temperingInfo["2-level DART"],
        nChainInfo["2-level DART"], init, nTuning, 0.3
    )

    dartFactory3 = DARTFactory("pcn")
    dartFactory3.target = tgtDensity
    dartFactory3.surrogate = [tgtDensity, tgtDensity]
    dart3 = build_dart(
        tgtDensity, dartFactory3,
        gammaInfo["3-level DART"], temperingInfo["3-level DART"],
        nChainInfo["3-level DART"], init, nTuning, 0.3
    )

    methods = {
        "MRW": mrw,
        "MALA": mala,
        "2-level DART": dart2,
        "3-level DART": dart3,
    }

    traces = {}

    print(
        f"\nRunning experiment: separation = {separation}, modeVar = {modeVar}")
    for name, _ in methodSpecs:
        print(f"  Running {name}...")
        method = methods[name]
        method.run(nSteps, initState.clone(), True)
        traces[name] = np.array(method.chain.trajectory).flatten()

    print("  Sampling finished.")

    lb = -0.5 * separation - 4.0 * np.sqrt(modeVar)
    rb = 0.5 * separation + 4.0 * np.sqrt(modeVar)
    grid = np.linspace(lb, rb, 400)

    logTgt = np.array([tgtDensity.evaluate_log(Scalar(x)) for x in grid])
    tgtEval = np.exp(logTgt - logTgt.max())
    tgtEval /= np.trapezoid(tgtEval, grid)

    # Compute surrogates for all levels (location at 0)
    surrogateEvals = {}
    allThetas = sorted(set(t for ts in temperingInfo.values() if ts is not None for t in ts))
    for theta in allThetas:
        surr = LocalisedSurrogateDensity(
            regularisation=gamma,
            tempering=theta,
            surrogateDensity=tgtDensity
        )
        surr.location = Scalar(0.0)
        logVals = np.array([surr.evaluate_log(Scalar(x)) for x in grid])
        vals = np.exp(logVals - logVals.max())
        vals /= np.trapezoid(vals, grid)
        surrogateEvals[theta] = vals

    densityMax = tgtEval.max()
    if surrogateEvals:
        densityMax = max(densityMax, max(e.max() for e in surrogateEvals.values()))

    return {
        "separation": separation,
        "modeVar": modeVar,
        "lb": lb,
        "rb": rb,
        "grid": grid,
        "tgtEval": tgtEval,
        "surrogateEvals": surrogateEvals,
        "traces": traces,
        "densityMax": densityMax,
    }


# =========================================================================
# Run both experiments
# =========================================================================
CACHE = Path(__file__).parent / 'joblib_caches' / f"{Path(__file__).stem}.joblib"
if hasJoblib:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
FORCE = False

PARAMS = dict(
    experiments=experiments,
    randomSeed=randomSeed,
    nTuning=nTuning,
    nSteps=nSteps,
    burnin=burnin,
)

results = None
if hasJoblib and CACHE.exists() and not FORCE:
    data = joblib.load(CACHE)
    if data.get('params') == PARAMS:
        results = data['results']
        print("Loaded results from cache.")
    else:
        print("Stale cache detected, recomputing...")

if results is None:
    if not hasJoblib:
        print("Warning: joblib is not installed. Caching is disabled.")
    results = [run_experiment(exp["separation"], exp["modeVar"], exp["gamma"])
               for exp in experiments]
    if hasJoblib:
        data = {'params': PARAMS, 'results': results}
        joblib.dump(data, CACHE, compress=3)


# =========================================================================
# Plot settings
# =========================================================================
if hasMatplotlib:
    plt.close('all')
    plt.style.use(Path(__file__).parent / 'manuscript.mplstyle')
    plt.rcParams.update({"axes.titlesize": 1, "lines.linewidth": 0.75})

    nMethods = len(methodSpecs)

    fig = plt.figure(figsize=(18.5, 1.935 * nMethods))
    gs = fig.add_gridspec(
        nMethods,
        4,
        width_ratios=[1.0, 0.16, 1.0, 0.16],
        wspace=0.10,
        hspace=0.22,
    )

    traceAxes = [[None, None] for _ in range(nMethods)]
    densAxes = [[None, None] for _ in range(nMethods)]

    for expIdx, result in enumerate(results):
        c0 = 2 * expIdx

        for i, (name, _) in enumerate(methodSpecs):
            trace = result["traces"][name]
            colour = colours[name]
            iat = ac.integrated_autocorrelation(trace[burnin:])
            iatInt = int(np.rint(iat))
            nCross = count_crossings(trace[burnin: burnin + plotSteps])

            ax = fig.add_subplot(
                gs[i, c0],
                sharex=traceAxes[0][expIdx] if i > 0 else None)
            traceAxes[i][expIdx] = ax

            ax.plot(trace[burnin: burnin + plotSteps],
                    linewidth=0.45, color=colour, alpha=0.85)
            ax.set_ylim(result["lb"], result["rb"])

            ax.text(
                0.005, 0.95, name,
                transform=ax.transAxes,
                fontsize=18,
                fontweight="bold",
                va="top",
                ha="left",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
            )

            ax.text(
                0.995, 0.95, f"IAT {iatInt}\n{nCross} crossings",
                transform=ax.transAxes,
                fontsize=15,
                fontweight="bold",
                va="top",
                ha="right",
                multialignment="right",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
            )

            if i < nMethods - 1:
                ax.tick_params(labelbottom=False)

            densAx = fig.add_subplot(gs[i, c0 + 1], sharey=ax)
            densAxes[i][expIdx] = densAx

            densAx.plot(
                result["tgtEval"], result["grid"],
                linewidth=1.5, color="0.3"
            )
            densAx.fill_betweenx(
                result["grid"],
                result["tgtEval"],
                alpha=0.06, color="0.3")

            tInfo = temperingInfo[name]
            if tInfo is not None:
                nLev = len(tInfo)
                for lvl, theta in enumerate(tInfo):
                    e = result["surrogateEvals"][theta]
                    frac = (lvl + 1) / nLev
                    densAx.plot(
                        e, result["grid"],
                        linewidth=1.5,
                        color=colour,
                        linestyle="--",
                        alpha=0.45 + 0.4 * frac,
                    )
                    densAx.fill_betweenx(
                        result["grid"],
                        e,
                        alpha=0.06 + 0.08 * frac,
                        color=colour,
                    )

            densAx.set_xlim(0, 1.05 * result["densityMax"])
            densAx.set_xticks([])
            densAx.tick_params(left=False, labelleft=False)
            for side in ["top", "right", "bottom", "left"]:
                densAx.spines[side].set_visible(False)

    traceAxes[-1][0].set_xlabel("iteration")
    traceAxes[-1][1].set_xlabel("iteration")

    fig.subplots_adjust(left=0.05, right=0.985, top=0.91, bottom=0.08)

    figuresDir = Path(__file__).parent.parent / "figures"
    figuresDir.mkdir(parents=True, exist_ok=True)
    figPath = figuresDir / "manuscript_multimodal.pdf"
    fig.savefig(figPath, bbox_inches="tight")
    print(f"\nSaved {figPath}")
    plt.close(fig)
else:
    print("Warning: matplotlib is not installed. Skipping plotting.")
