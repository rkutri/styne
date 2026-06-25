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
neloFactory = DARTFactory(root="mrw")
neloFactory.target = tgtDensity
neloFactory.surrogate = [srrDensity]
neloFactory.regularisation = [1e-10]
neloFactory.nChain = [0]
neloFactory.root.proposalCovariance = IIDCovarianceMatrix(DIM, 5.)

# Seed the factories
rngSeed = 12345
neloFactory.rng = default_rng(rngSeed)

neloMCMC = neloFactory.create()

# Override the initial measure to be a Gaussian centered at the current state,
# so that subchain length 0 recovers the MRW proposal rather than the current state.
proposalCovariance = IIDCovarianceMatrix(DIM, 5.)
neloMCMC.proposal.measure._initialMeasure = Gaussian(proposalCovariance)

# create MRW with the corresponding settings
mrwFactory = MRWFactory()
mrwFactory.target = tgtDensity
mrwFactory.proposalCovariance = IIDCovarianceMatrix(DIM, 5.)
mrwFactory.rng = default_rng(rngSeed)

mrwMCMC = mrwFactory.create()

nSteps = int(1e5)
initState = ScalarParameter(-3.)

neloMCMC.run(nSteps, initState, False)
mrwMCMC.run(nSteps, initState, False)

neloStates = np.array(neloMCMC.chain.trajectory)
mrwStates = np.array(mrwMCMC.chain.trajectory)

# postprocessing
burnin = int(0.02 * nSteps)

assert nSteps > burnin

# estimate autocorrelation function
neloIAT = ac.sokal_heuristic(
    ac.estimate_autocorrelation_function_1d(np.squeeze(neloStates[burnin:])), 5.
)
mrwIAT = ac.sokal_heuristic(
    ac.estimate_autocorrelation_function_1d(np.squeeze(mrwStates[burnin:])), 5.
)

assert neloIAT == mrwIAT

neloSamples = neloStates[burnin::neloIAT]
mrwSamples = mrwStates[burnin::mrwIAT]

# estimate mean
neloMean = np.mean(neloSamples, axis=0)
mrwMean = np.mean(mrwSamples, axis=0)
assert np.allclose(neloMean, mrwMean, rtol=RTOL, atol=ATOL)

neloVar = np.var(neloSamples, axis=0)
mrwVar = np.var(mrwSamples, axis=0)
assert np.allclose(neloVar, mrwVar, rtol=RTOL, atol=ATOL)

neloAccept = neloMCMC.diagnostics.global_acceptance_rate()
mrwAccept = mrwMCMC.diagnostics.global_acceptance_rate()
assert np.isclose(neloAccept, mrwAccept, rtol=RTOL, atol=ATOL)
