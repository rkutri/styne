import numpy as np
from scipy.signal import correlate
from styne.mcmc.chain import Chain


def estimate_autocorrelation_function_1d(sequence):
    """
    Return the biased normalized autocorrelation function for lags 0..n-1.
    """
    sequence = np.asarray(sequence)
    if sequence.ndim != 1 or sequence.size == 0:
        return np.array([])
    n = sequence.size
    centered = sequence - np.mean(sequence)
    acf = correlate(centered, centered, mode="full", method="auto")[n - 1:]
    if acf.size == 0 or acf[0] == 0:
        return np.zeros_like(acf)
    return acf / acf[0]


def sokal_heuristic(tauSeq, heuristicConst):
    """
    Return truncation lag t* (1-based) using Sokal's rule t > c * tau(t).
    """
    if len(tauSeq) == 0 or not np.all(np.isfinite(tauSeq)):
        return 0
    t = np.arange(1, len(tauSeq) + 1)
    condition = t > (heuristicConst * tauSeq)
    idx = np.nonzero(condition)[0]
    if idx.size > 0:
        truncLag = int(idx[0] + 1)
    else:
        truncLag = len(tauSeq)
    return max(1, min(truncLag, len(tauSeq)))


def integrated_autocorrelation_1d(acf, sokalConst=5.0):
    """
    Estimate the integrated autocorrelation time (IAT) tau from a normalized
    autocorrelation function using the method described in Goodman & Weare
    (2010) and implemented in the emcee sampler
    (https://github.com/dfm/emcee). The procedure follows Sokal's
    self-consistent windowing rule:
        tau(t) = 1 + 2 * sum_{k=1..t} acf[k]
        choose t* such that t* > c * tau(t*),
    and return tau(t*).

    Non-closing fix: when the rule is never satisfied, sokal_heuristic
    returns the full length len(tauSeq). The value tauSeq[-1] there is ~0,
    because the centred ACF satisfies sum_{k>=1} acf[k] = -1/2, which would
    send ESS = 1/tau to infinity. In that case the chain is too short to
    resolve tau, which is at least ~ N / sokalConst, so we return that lower
    bound rather than ~0.

    Returns np.nan on invalid input.
    """
    acf = np.atleast_1d(acf)
    if acf.ndim != 1 or acf.size < 2:
        return np.nan
    acfLags = acf[1:]
    if acfLags.size == 0:
        return np.nan
    cumsumAcf = np.cumsum(acfLags, dtype=float)
    tauSeq = 1.0 + 2.0 * cumsumAcf
    truncLag = sokal_heuristic(tauSeq, sokalConst)
    if truncLag <= 0 or truncLag >= len(tauSeq):
        # Window did not close (or closed only at the final, ~0 lag).
        return float(len(tauSeq) / sokalConst)
    tau = float(tauSeq[truncLag - 1])
    return tau if (np.isfinite(tau) and tau > 0) else np.nan


def integrated_autocorrelation(seq, method="mean", sokalConst=5.0):
    """
    Wrapper estimating tau for 1D or 2D sequences.
    method: 'mean' (ACF of mean across dims) or 'max' (max tau across dims).
    Returns np.nan on invalid input.
    """
    seq = np.asarray(seq)
    if seq.size == 0:
        return np.nan
    if seq.ndim == 1:
        acf = estimate_autocorrelation_function_1d(seq)
        return integrated_autocorrelation_1d(acf, sokalConst=sokalConst)
    if seq.ndim >= 2:
        if method not in ("mean", "max"):
            return np.nan
        if method == "mean":
            try:
                meanSeries = np.mean(seq, axis=1)
            except Exception:
                return np.nan
            acf = estimate_autocorrelation_function_1d(meanSeries)
            return integrated_autocorrelation_1d(acf, sokalConst=sokalConst)
        try:
            dimensionCount = seq.shape[1]
        except Exception:
            return np.nan
        iatList = []
        for d in range(dimensionCount):
            dimensionSequence = seq[:, d]
            dimensionAcf = estimate_autocorrelation_function_1d(dimensionSequence)
            dimensionTau = integrated_autocorrelation_1d(dimensionAcf, sokalConst=sokalConst)
            iatList.append(dimensionTau if np.isfinite(dimensionTau) else np.nan)
        finiteVals = [v for v in iatList if np.isfinite(v)]
        return float(max(finiteVals)) if finiteVals else np.nan
    return np.nan


def multichain_ess_per_iter(chains):
    """
    Effective sample size per iteration of a scalar quantity from M chains.

    Combines the per-chain autocorrelation (via
    'estimate_autocorrelation_function_1d') with the between-chain variance
    using the Gelman/Vehtari multi-chain estimator, summed by Geyer's initial
    positive sequence. It is smooth where the quantity mixes and returns 1/N
    for a frozen quantity (zero within-chain variance) rather than inverting
    to a spurious 1. Feed it the slowest-direction projection of several
    replicate chains run from overdispersed starts.

    Parameters
    ----------
    chains : array_like, shape (M, N)
        M chains of a single scalar quantity, each of length N.

    Returns
    -------
    float
        Effective samples per iteration, in (0, 1]; np.nan if N < 2.
    """
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    nChains, nSamples = chains.shape
    if nSamples < 2:
        return np.nan

    withinVariance = float(chains.var(axis=1, ddof=1).mean())
    if not np.isfinite(withinVariance) or withinVariance <= 0.0:    # all chains frozen flat
        return 1.0 / nSamples

    means = chains.mean(axis=1)
    betweenVariance = nSamples * float(means.var(ddof=1)) if nChains > 1 else 0.0
    variancePlus = (nSamples - 1) / nSamples * withinVariance + betweenVariance / nSamples

    biasedVariance = chains.var(axis=1)                        # ddof=0, equals gamma_m(0)
    pooledAutocovariance = np.zeros(nSamples)
    for m in range(nChains):
        acf = estimate_autocorrelation_function_1d(chains[m])
        pooledAutocovariance[:acf.size] += acf * biasedVariance[m]
    pooledAutocovariance /= nChains

    rho = 1.0 - (withinVariance - pooledAutocovariance) / variancePlus
    tau = 1.0
    for k in range(1, nSamples // 2):
        pair = rho[2 * k - 1] + rho[2 * k]
        if pair < 0.0:
            break
        tau += 2.0 * pair
    return 1.0 / max(tau, 1.0)
