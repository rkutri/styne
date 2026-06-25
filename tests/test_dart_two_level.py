import numpy as np
import styne.utility.postprocessing as ac

from numpy.random import default_rng
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import Gaussian, GaussianDensity
from styne.mcmc.method.dart import DARTFactory
from styne.mcmc.method.mrw import MRWFactory
from styne.parameter.scalar import Scalar as ScalarParameter
from styne.statistics.interface import DensityInterface


class ZeroSurrogateDensity(DensityInterface):
    """Surrogate density that is identically zero."""

    @property
    def domainType(self):
        return ScalarParameter

    @property
    def domainDimension(self) -> int:
        return 1

    def evaluate_log(self, parameter) -> float:
        return 0.0


# Enable Gaussian to act as an initial measure by adding a location property
Gaussian.location = property(
    fget=lambda self: self.mean,
    fset=lambda self, value: setattr(self, "mean", value)
)


# LIMIT 1: METROPOLIS-HASTINGS
# ----------------------------

# The special case of subchain length zero recovers standard Metropolis-Hastings
# The (Gaussian) initial measure then corresponds to the proposal measure
# and the regularisation variance is irrelevant.

DIM = 1
RTOL = 1e-1
ATOL = 1e-2

# target density
tgtMean = ScalarParameter(1.5)
tgtVar = 1.
tgtDensity = GaussianDensity(IIDCovarianceMatrix(DIM, tgtVar), tgtMean)


srrDensity = ZeroSurrogateDensity()

# DART configuration
dartFactory = DARTFactory(root="mrw")
dartFactory.target = tgtDensity
dartFactory.surrogate = [srrDensity]
dartFactory.regularisation = [1e-10]
dartFactory.nChain = [0]
dartFactory.root.proposalCovariance = IIDCovarianceMatrix(DIM, 5.)

# Seed the factories
rngSeed = 12345
dartFactory.rng = default_rng(rngSeed)

dartMCMC = dartFactory.create()

# Override the initial measure to be a Gaussian centered at the current state,
# so that subchain length 0 recovers the MRW proposal rather than the current state.
proposalCovariance = IIDCovarianceMatrix(DIM, 5.)
dartMCMC.proposal.measure._initialMeasure = Gaussian(proposalCovariance)

# create MRW with the corresponding settings
mrwFactory = MRWFactory()
mrwFactory.target = tgtDensity
mrwFactory.proposalCovariance = IIDCovarianceMatrix(DIM, 5.)
mrwFactory.rng = default_rng(rngSeed)

mrwMCMC = mrwFactory.create()

nSteps = int(1e5)
initState = ScalarParameter(-3.)

dartMCMC.run(nSteps, initState, False)
mrwMCMC.run(nSteps, initState, False)

dartStates = np.array(dartMCMC.chain.trajectory)
mrwStates = np.array(mrwMCMC.chain.trajectory)

# postprocessing
burnin = int(0.02 * nSteps)

assert nSteps > burnin

# estimate autocorrelation function
dartIAT = ac.sokal_heuristic(
    ac.estimate_autocorrelation_function_1d(np.squeeze(dartStates[burnin:])), 5.
)
mrwIAT = ac.sokal_heuristic(
    ac.estimate_autocorrelation_function_1d(np.squeeze(mrwStates[burnin:])), 5.
)

assert dartIAT == mrwIAT

dartSamples = dartStates[burnin::dartIAT]
mrwSamples = mrwStates[burnin::mrwIAT]

# estimate mean
dartMean = np.mean(dartSamples, axis=0)
mrwMean = np.mean(mrwSamples, axis=0)
assert np.allclose(dartMean, mrwMean, rtol=RTOL, atol=ATOL)

dartVar = np.var(dartSamples, axis=0)
mrwVar = np.var(mrwSamples, axis=0)
assert np.allclose(dartVar, mrwVar, rtol=RTOL, atol=ATOL)

dartAccept = dartMCMC.diagnostics.global_acceptance_rate()
mrwAccept = mrwMCMC.diagnostics.global_acceptance_rate()
assert np.isclose(dartAccept, mrwAccept, rtol=RTOL, atol=ATOL)
