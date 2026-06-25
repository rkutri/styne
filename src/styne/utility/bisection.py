import math
from typing import Callable, Optional


def bisection(
    f: Callable[[float], float],
    fTgt: float,
    xLo: float,
    xHi: float,
    log: bool = True,
    xLimitLo: Optional[float] = None,
    xLimitHi: Optional[float] = None,
    tol: float = 0.05,
    maxIter: int = 10,
    maxExpansions: int = 12,
    expandFactor: float = 10.,
    yLo: Optional[float] = None,
    yHi: Optional[float] = None,
):
    """
    Bisection search for x in [xLo, xHi] such that f(x) ~= fTgt (within tol).

    Parameters
    ----------
    f            : monotone callable
    fTgt         : target value
    xLo, xHi    : initial bracket (xLo < xHi); must lie within [xLimitLo, xLimitHi]
    log          : work in log-space (x must be positive)
    xLimitLo/Hi : hard bounds; f is never evaluated outside them
    tol          : convergence criterion on |f(x) - fTgt|
    maxIter      : bisection iterations after bracketing
    maxExpansions: expansion attempts when initial bracket is not valid
    expandFactor : multiplicative step for bracket expansion
    yLo, yHi    : pre-computed f(xLo) - fTgt and f(xHi) - fTgt; if supplied
                  the corresponding endpoint evaluations are skipped

    Expansion strategy
    ------------------
    When both endpoints are on the same side of fTgt the bracket is expanded
    toward the endpoint closer to fTgt.  After each one-sided expansion the
    opposite endpoint is *tightened* to the previous boundary: it already has
    the correct sign and is a strictly tighter bound than the original endpoint.
    This avoids wasted evaluations at the wide original bound once a bracket is
    found.
    """

    # ── validation ──────────────────────────────────────────────────────────
    if expandFactor <= 1.0:
        raise ValueError("expandFactor must be > 1.0")
    if maxExpansions < 0 or maxIter < 0:
        raise ValueError("maxExpansions and maxIter must be non-negative")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if xLo >= xHi:
        raise ValueError("xLo must be < xHi")

    if log:
        if xLo <= 0 or xHi <= 0:
            raise ValueError("xLo and xHi must be positive when log=True")
        if xLimitLo is not None and xLimitLo <= 0:
            raise ValueError("xLimitLo must be positive when log=True")
        if xLimitHi is not None and xLimitHi <= 0:
            raise ValueError("xLimitHi must be positive when log=True")

    if xLimitLo is not None and xLo < xLimitLo:
        raise ValueError("xLo < xLimitLo")
    if xLimitHi is not None and xHi > xLimitHi:
        raise ValueError("xHi > xLimitHi")

    EPS = 1e-15

    def eval_f(x: float) -> float:
        if xLimitLo is not None and x < xLimitLo - EPS:
            raise RuntimeError("Attempt to evaluate f(x) below xLimitLo")
        if xLimitHi is not None and x > xLimitHi + EPS:
            raise RuntimeError("Attempt to evaluate f(x) above xLimitHi")
        y = f(x) - fTgt
        if not math.isfinite(y):
            raise RuntimeError(f"Non-finite f(x) encountered at x={x}")
        return y

    def clamp(x, lo, hi):
        if lo is not None:
            x = max(x, lo)
        if hi is not None:
            x = min(x, hi)
        return x

    def at_lo_limit(x):
        return xLimitLo is not None and x <= xLimitLo + EPS

    def at_hi_limit(x):
        return xLimitHi is not None and x >= xLimitHi - EPS

    # ── initial evaluation (skip if pre-computed values supplied) ────────────
    fLo = yLo if yLo is not None else eval_f(xLo)
    fHi = yHi if yHi is not None else eval_f(xHi)

    if abs(fLo) <= tol:
        return xLo
    if abs(fHi) <= tol:
        return xHi

    # ── bracket expansion ────────────────────────────────────────────────────
    for _ in range(maxExpansions):
        if fLo * fHi <= 0:
            break

        expandBoth = abs(fLo) == abs(fHi)
        expandLo = abs(fLo) <= abs(fHi)

        if expandBoth:
            xLoNew = clamp(xLo / expandFactor, xLimitLo, None)
            xHiNew = clamp(xHi * expandFactor, None, xLimitHi)
            if at_lo_limit(xLoNew) or at_hi_limit(xHiNew):
                xLo, xHi = xLoNew, xHiNew
                break
            xLo, xHi = xLoNew, xHiNew
            fLo = eval_f(xLo)
            fHi = eval_f(xHi)

        elif expandLo:
            xLoNew = clamp(xLo / expandFactor, xLimitLo, None)
            # Tighten upper end to old xLo (same sign, strictly tighter)
            xHi, fHi = xLo, fLo
            if at_lo_limit(xLoNew):
                xLo = xLoNew
                break
            xLo = xLoNew
            fLo = eval_f(xLo)

        else:  # expandHi
            xHiNew = clamp(xHi * expandFactor, None, xLimitHi)
            # Tighten lower end to old xHi (same sign, strictly tighter)
            xLo, fLo = xHi, fHi
            if at_hi_limit(xHiNew):
                xHi = xHiNew
                break
            xHi = xHiNew
            fHi = eval_f(xHi)

        if abs(fLo) <= tol:
            return xLo
        if abs(fHi) <= tol:
            return xHi

    if fLo * fHi > 0:
        raise RuntimeError(
            f"Could not bracket target within {maxExpansions} expansions."
        )

    # ── bisection ────────────────────────────────────────────────────────────
    lo = math.log(xLo) if log else xLo
    hi = math.log(xHi) if log else xHi

    for _ in range(maxIter):
        mid = 0.5 * (lo + hi)
        xMid = math.exp(mid) if log else mid

        # Safety clamp (shouldn't trigger with well-formed limits)
        if xLimitLo is not None and xMid < xLimitLo:
            xMid = xLimitLo
            mid = math.log(xMid) if log else xMid
        if xLimitHi is not None and xMid > xLimitHi:
            xMid = xLimitHi
            mid = math.log(xMid) if log else xMid

        fMid = eval_f(xMid)

        if abs(fMid) <= tol:
            return xMid
        if mid == lo or mid == hi:
            break

        if fLo * fMid <= 0:
            hi, fHi = mid, fMid
        else:
            lo, fLo = mid, fMid

    return math.exp(0.5 * (lo + hi)) if log else 0.5 * (lo + hi)
