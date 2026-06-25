"""
Consolidated MCMC tuning module.

Provides:
  - Low-level bisection primitives (estimate_acceptance, tune_parameter)
  - infer_init(target) helper
  - Config dataclasses (RWTunerConfig, LangevinTunerConfig)
  - Per-method tuner classes (MRWTuner, MALATuner, PCNTuner, PMALATuner)
"""

import logging
import numpy as np
from dataclasses import dataclass

from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import GaussianDensity
from styne.statistics.radonnikodym import RadonNikodym
from styne.utility.bisection import bisection

logger = logging.getLogger(__name__)


# ─── Low-level primitives ────────────────────────────────────────────────────

def estimate_acceptance(
    factory, configure_factory, init, nTuning, optValue
) -> float:

    logger.debug(f"      * probing {type(factory).__name__[:-7]} with value: {optValue}")

    configure_factory(factory, optValue)
    mcmc = factory.create()
    mcmc.run(nTuning, init.clone())

    logger.debug(
        "        -> resulting acceptance: "
        f"{mcmc.diagnostics.global_acceptance_rate()}"
    )

    return mcmc.diagnostics.global_acceptance_rate()


def tune_parameter(
    objective,
    acceptanceTarget,
    xLo: float,
    xHi: float,
    log: bool = True,
    xLimitLo=None,
    xLimitHi=None,
    tolerance: float = 0.05,
    maxIter: int = 12,
    accLo: float = None,
    accHi: float = None,
):
    return bisection(
        objective,
        acceptanceTarget,
        xLo, xHi,
        log,
        xLimitLo, xLimitHi,
        tol=tolerance,
        maxIter=maxIter,
        yLo=None if accLo is None else accLo - acceptanceTarget,
        yHi=None if accHi is None else accHi - acceptanceTarget,
    )


# ─── Config dataclasses ──────────────────────────────────────────────────────

@dataclass
class RWTunerConfig:
    acceptanceGoal: float = 0.3
    nTuning: int = 1000
    tolerance: float = 0.05

@dataclass
class LangevinTunerConfig:
    acceptanceGoal: float = 0.6
    nTuning: int = 1000
    tolerance: float = 0.05




# ─── Init helper ─────────────────────────────────────────────────────────────

def infer_init(target):
    """Return a sensible starting point for the given target density."""
    if isinstance(target, RadonNikodym):
        return target.reference.mean.clone()
    elif isinstance(target, GaussianDensity):
        return target.mean.clone()
    else:
        return target.domainType(np.zeros(target.domainDimension))


# ─── Per-method tuner classes ────────────────────────────────────────────────

class MRWTuner:
    """Tunes an MRW sampler by bisecting on proposal variance."""

    def __init__(self, factory, init, config=None):
        self._factory = factory
        self._init = init
        self._config = config if config is not None else RWTunerConfig()

    def tune(self):
        dim = self._factory.target.domainDimension
        state = {'probes': 0, 'acc': 0.0}

        def objective(pv):
            self._factory.proposalCovariance = IIDCovarianceMatrix(dim, pv)
            mcmc = self._factory.create()
            mcmc.run(self._config.nTuning, self._init.clone())
            acc = mcmc.diagnostics.global_acceptance_rate()
            logger.debug(f"      * probing MRW propVar={pv:.3e}: acc={acc:.3f}")
            state['probes'] += 1
            state['acc'] = acc
            return acc

        pvOpt = tune_parameter(
            objective, self._config.acceptanceGoal, xLo=1e-9, xHi=1e-1, tolerance=self._config.tolerance
        )
        self._factory.proposalCovariance = IIDCovarianceMatrix(dim, pvOpt)
        logger.info(f"MRW tuned  propVar={pvOpt:.1e}  (acc {state['acc']:.2f}, {state['probes']} probes)")
        return self._factory.create()


class MALATuner:
    """Tunes a MALA sampler by bisecting on step size."""

    def __init__(self, factory, init, config=None):
        self._factory = factory
        self._init = init
        self._config = config if config is not None else LangevinTunerConfig()

    def tune(self):
        state = {'probes': 0, 'acc': 0.0}

        def objective(h):
            self._factory.stepSize = h
            mcmc = self._factory.create()
            mcmc.run(self._config.nTuning, self._init.clone())
            acc = mcmc.diagnostics.global_acceptance_rate()
            logger.debug(f"      * probing MALA stepSize={h:.3e}: acc={acc:.3f}")
            state['probes'] += 1
            state['acc'] = acc
            return acc

        hOpt = tune_parameter(
            objective, self._config.acceptanceGoal, xLo=1e-9, xHi=1e-0,
            tolerance=self._config.tolerance
        )
        self._factory.stepSize = hOpt
        logger.info(f"MALA tuned  stepSize={hOpt:.1e}  (acc {state['acc']:.2f}, {state['probes']} probes)")
        return self._factory.create()


class PCNTuner:
    """Tunes a pCN sampler by bisecting on beta."""

    def __init__(self, factory, init, config=None):
        self._factory = factory
        self._init = init
        self._config = config if config is not None else RWTunerConfig()

    def tune(self):
        state = {'probes': 0, 'acc': 0.0}

        def objective(b):
            self._factory.beta = b
            mcmc = self._factory.create()
            mcmc.run(self._config.nTuning, self._init.clone())
            acc = mcmc.diagnostics.global_acceptance_rate()
            logger.debug(f"      * probing pCN beta={b:.4f}: acc={acc:.3f}")
            state['probes'] += 1
            state['acc'] = acc
            return acc

        betaOpt = tune_parameter(
            objective,
            self._config.acceptanceGoal,
            xLo=1e-5,
            xHi=1.0,
            xLimitLo=1e-6,
            xLimitHi=1.0,
            log=False,
            tolerance=self._config.tolerance,
        )
        self._factory.beta = betaOpt
        logger.info(f"pCN tuned  beta={betaOpt:.3f}  (acc {state['acc']:.2f}, {state['probes']} probes)")
        return self._factory.create()


class PMALATuner:
    """Tunes a pMALA sampler by bisecting on beta."""

    def __init__(self, factory, init, config=None):
        self._factory = factory
        self._init = init
        self._config = config if config is not None else LangevinTunerConfig()

    def tune(self):
        state = {'probes': 0, 'acc': 0.0}

        def objective(b):
            self._factory.beta = b
            mcmc = self._factory.create()
            mcmc.run(self._config.nTuning, self._init.clone())
            acc = mcmc.diagnostics.global_acceptance_rate()
            logger.debug(f"      * probing pMALA beta={b:.4f}: acc={acc:.3f}")
            state['probes'] += 1
            state['acc'] = acc
            return acc

        betaOpt = tune_parameter(
            objective,
            self._config.acceptanceGoal,
            xLo=1e-2,
            xHi=1.0,
            xLimitLo=1e-6,
            xLimitHi=1.0,
            log=False,
            tolerance=self._config.tolerance,
        )
        self._factory.beta = betaOpt
        logger.info(f"pMALA tuned  beta={betaOpt:.3f}  (acc {state['acc']:.2f}, {state['probes']} probes)")
        return self._factory.create()
