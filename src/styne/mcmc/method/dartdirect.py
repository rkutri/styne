import numpy as np
from typing import Optional
from numpy.random import Generator

from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.transition import TransitionData
from styne.mcmc.diagnostics import ChainDiagnostics
from styne.mcmc.acceptance import AcceptanceProbability
from styne.parameter.parameter import Parameter
from styne.statistics.gaussian import Gaussian
from styne.statistics.interface import DensityInterface
from styne.statistics.covariance import DenseCovarianceMatrix


class DirectDARTProposal(ProposalMethod):
    """
    Exact Gaussian proposal for the DART sampler.

    Given a surrogate Gaussian (the Laplace approximation at the MAP), a tempering 
    parameter theta, and a regularisation parameter gamma, the proposal draws from 
    N(mu_x, P^{-1}), where:
        P = theta * A + gamma * I
        mu_x = P^{-1} b_x
        b_x = theta * A * x_hat + gamma * x

    Here, x_hat and A are the mean and precision (inverse covariance) of the 
    surrogate Gaussian.

    Parameters
    ----------
    tempering : float
        Tempering parameter theta in (0, 1].
    gamma : float
        Regularisation strength gamma > 0.
    surrogate : Gaussian
        Gaussian representing the Laplace approximation.
    proposalCovariance : DenseCovarianceMatrix
        The precomputed covariance of the proposal, P^{-1}.
    """

    def __init__(self, tempering: float, gamma: float, surrogate: Gaussian,
                 proposalCovariance: DenseCovarianceMatrix):
        super().__init__()
        self._tempering = tempering
        self._gamma = gamma
        self._surrogate = surrogate
        
        # Precompute A * x_hat (where A is surrogate precision).
        # We can apply the precision matrix A via apply_inverse on the surrogate covariance.
        self._aTimesXhat = self._surrogate.covariance.apply_inverse(self._surrogate.mean.coordinate)
        
        self._proposalMeasure = Gaussian(proposalCovariance)

    def generate_proposal(self, rng: Generator) -> TransitionData:
        if self._state is None:
            raise ValueError("Trying to generate proposal with undefined state")

        x = self._state.coordinate
        bx = self._tempering * self._aTimesXhat + self._gamma * x
        
        # Apply P^{-1} to get mu_x
        mux = self._proposalMeasure.covariance.apply(bx)
        
        propMean = self._state.clone()
        propMean.coordinate = mux
        self._proposalMeasure.mean = propMean
        
        proposal = self._proposalMeasure.generate_realisation(rng=rng)
        
        return TransitionData(
            self._state, proposal, auxiliary={'bx': bx, 'mux': mux}
        )


class DirectDART(MetropolisHastings):
    """
    DART (Data-Assimilation based Regularised Transition) sampler.

    Exact implementation of the DART log-acceptance ratio using a Gaussian 
    surrogate for the local structure. 

    Parameters
    ----------
    targetDensity : DensityInterface
        Target density to sample from.
    tempering : float
        Tempering parameter theta.
    gamma : float
        Regularisation strength gamma.
    surrogate : Gaussian
        Gaussian representing the Laplace approximation at the MAP.
    proposalCovariance : DenseCovarianceMatrix
        Precomputed covariance matrix P^{-1} of the proposal.
    diagnostics : ChainDiagnostics
        Tracks transition statistics.
    acceptance : AcceptanceProbability, optional
        Acceptance rule (defaults to standard Metropolis-Hastings).
    """

    def __init__(self, targetDensity: DensityInterface, tempering: float,
                 gamma: float, surrogate: Gaussian, 
                 proposalCovariance: DenseCovarianceMatrix, 
                 diagnostics: ChainDiagnostics,
                 acceptance: AcceptanceProbability = None,
                 rng: Optional[Generator] = None):
        
        if tempering <= 0.0 or tempering > 1.0:
            raise ValueError("Tempering parameter must be in (0, 1].")
        if gamma <= 0.0:
            raise ValueError("Regularisation gamma must be strictly positive.")

        self._tempering = tempering
        self._gamma = gamma
        self._surrogate = surrogate

        proposalMethod = DirectDARTProposal(tempering, gamma, surrogate, proposalCovariance)
        super().__init__(targetDensity, proposalMethod, diagnostics, acceptance=acceptance, rng=rng)

    def _log_mh_ratio(self, transition: TransitionData) -> float:
        
        x = transition.state.coordinate
        z = transition.proposal.coordinate

        # 1. Target density difference
        logTargetDiff = float(
            self._tgtDensity.evaluate_log(transition.proposal)
            - self._tgtDensity.evaluate_log(transition.state)
        )
        if logTargetDiff == -np.inf:
            return -np.inf

        xHat = self._surrogate.mean.coordinate

        # 2. Quadratic surrogate correction
        # We need: (theta / 2) * ( (z - xHat)^T A (z - xHat) - (x - xHat)^T A (x - xHat) )
        # A is the precision of the surrogate. We use dual_quadratic_form on surrogate covariance.
        diffZ = z - xHat
        diffX = x - xHat
        quadDiffZ = self._surrogate.covariance.dual_quadratic_form(diffZ)
        quadDiffX = self._surrogate.covariance.dual_quadratic_form(diffX)
        quadSurrogateDiff = 0.5 * self._tempering * (quadDiffZ - quadDiffX)

        # 3. Normalisation ratio log(N_x / N_z)
        bx = transition.auxiliary['bx']
        mux = transition.auxiliary['mux']
        
        # Need to compute b_z and mu_z for the reverse proposal
        bz = self._tempering * self._proposalMethod._aTimesXhat + self._gamma * z
        muz = self._proposalMethod._proposalMeasure.covariance.apply(bz)

        normDiff = 0.5 * (mux.dot(bx) - muz.dot(bz)) - 0.5 * self._gamma * (x.dot(x) - z.dot(z))

        return logTargetDiff + quadSurrogateDiff + normDiff
