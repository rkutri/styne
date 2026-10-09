from typing import Optional

from numpy.random import Generator

from styne.backend import infer_backend
from styne.parameter.parameter import Parameter
from styne.mcmc.diagnostics import ChainDiagnostics
from styne.mcmc.proposal import ProposalMethod, zero_correction
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.acceptance import AcceptanceProbability
from styne.mcmc.factory import MHFactory
from styne.statistics.interface import DensityInterface
from styne.statistics.covariance import CovarianceMatrix
from styne.statistics.gaussian import Gaussian


class MRWProposal(ProposalMethod):
    """
    Symmetric Gaussian random walk proposal.

    Draws proposals from N(state, proposalCov).

    Parameters
    ----------
    proposalCov : CovarianceMatrix
        Covariance of the proposal distribution.
    """

    def __init__(self, proposalCov: CovarianceMatrix):
        self._proposalMeasure = Gaussian(proposalCov)

    @property
    def covariance(self) -> CovarianceMatrix:
        """Current proposal covariance."""
        return self._proposalMeasure.covariance

    @covariance.setter
    def covariance(self, cov: CovarianceMatrix):
        """Replace the proposal covariance."""
        self._proposalMeasure = self._proposalMeasure.with_covariance(cov)

    def propose(self, state: Parameter, rng):
        return self.propose_with_covariance(state, self.covariance, rng)

    @staticmethod
    def propose_with_covariance(state, covariance, rng):
        backend = infer_backend(state.coordinate)
        metadata = backend.metadata(state.coordinate)
        noise, nextRng = backend.normal(
            rng, state.coordinate.shape,
            dtype=metadata.dtype, device=metadata.device,
        )
        proposal = state.with_coordinate(
            state.coordinate + covariance.apply_chol_factor(noise)
        )
        return ProposalMethod.record(
            state, proposal, zero_correction(state)
        ), nextRng


class MetropolisedRandomWalk(MetropolisHastings):
    """
    Metropolis-Hastings with a symmetric Gaussian random walk proposal.

    Since the proposal kernel is symmetric, the log MH ratio reduces to
    the log-density difference between proposal and current state.

    Parameters
    ----------
    target : DensityInterface
        Target density to sample from.
    proposalCov : CovarianceMatrix
        Covariance of the Gaussian proposal kernel.
    diagnostics : ChainDiagnostics
        Tracks transition statistics.
    """
    name = "MRW"

    def __init__(
            self,
            target: DensityInterface,
            proposalCov: CovarianceMatrix,
            diagnostics: ChainDiagnostics,
            acceptance: AcceptanceProbability = None,
            rng: Optional[Generator] = None):

        proposalMethod = MRWProposal(proposalCov)
        super().__init__(target, proposalMethod, diagnostics,
                         acceptance=acceptance, rng=rng)

    @property
    def proposalCovariance(self) -> CovarianceMatrix:
        """Current proposal covariance."""
        return self._proposalMethod.covariance

    @proposalCovariance.setter
    def proposalCovariance(self, cov: CovarianceMatrix):
        """Replace the proposal covariance."""
        self._proposalMethod.covariance = cov


class MRWFactory(MHFactory):
    """Factory for constructing 'MetropolisedRandomWalk' instances."""

    def __init__(self):
        super().__init__()
        self._proposalCov: CovarianceMatrix = None
        self._acceptance: AcceptanceProbability = None

    @property
    def proposalCovariance(self) -> CovarianceMatrix:
        return self._proposalCov

    @proposalCovariance.setter
    def proposalCovariance(self, cov: CovarianceMatrix):
        self._proposalCov = cov

    @property
    def acceptance(self) -> AcceptanceProbability:
        return self._acceptance

    @acceptance.setter
    def acceptance(self, strategy: AcceptanceProbability):
        self._acceptance = strategy

    def _validate(self) -> None:
        super()._validate()
        if self._proposalCov is None:
            raise ValueError("Proposal covariance not set for MRWFactory.")

    def _create_sampler(self) -> MetropolisHastings:
        return MetropolisedRandomWalk(
            self._target, self._proposalCov, self._diagnostics,
            self._acceptance, rng=self.rng)
