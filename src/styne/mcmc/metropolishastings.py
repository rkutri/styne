from abc import abstractmethod
from typing import Optional

from numpy import log
from numpy.random import Generator

from styne.mcmc.sampler import MCMCSampler
from styne.parameter.parameter import Parameter
from styne.statistics.interface import DensityInterface
from styne.mcmc.transition import TransitionData
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.chain import Chain
from styne.mcmc.annotator import Annotator
from styne.mcmc.diagnostics import ChainDiagnostics
from styne.mcmc.acceptance import AcceptanceProbability, StandardAcceptance


class MetropolisHastings(MCMCSampler):
    """Metropolis-Hastings sampler base class.

    Subclasses implement '_log_mh_ratio' to define the variant of the algorithm.
    This class handles accept/reject, diagnostics, and chain bookkeeping.

    Parameters
    ----------
    targetDensity : DensityInterface
        Target density to sample from.
    proposalMethod : ProposalMethod
        Proposal mechanism. Assembled by the concrete subclass constructor,
        not passed through by most subclasses' own public constructors.
    diagnostics : ChainDiagnostics
        Tracks transition statistics.
    acceptance : AcceptanceProbability, optional
        Defaults to 'StandardAcceptance' if not given.
    rng : Generator, optional
    """

    def __init__(
        self,
        targetDensity: DensityInterface,
        proposalMethod: ProposalMethod,
        diagnostics: ChainDiagnostics,
        acceptance: Optional[AcceptanceProbability] = None,
        rng: Optional[Generator] = None,
    ) -> None:

        super().__init__(rng=rng)

        self._tgtDensity = targetDensity
        self._proposalMethod = proposalMethod
        self._diagnostics = diagnostics
        self._acceptance = acceptance if acceptance is not None else StandardAcceptance()

        self._chain = Chain()
        self._annotator: Optional[Annotator] = None

    @property
    def chain(self):
        if not self._storeChain:
            raise RuntimeError(
                "Chain storage is disabled (storeChain=False)."
            )
        return self._chain

    @property
    def target(self):
        return self._tgtDensity

    @target.setter
    def target(self, targetDensity: DensityInterface):
        """Replace the target density and reset state."""
        self._tgtDensity = targetDensity
        self.clear()

    @property
    def proposal(self):
        return self._proposalMethod

    @property
    def acceptance(self) -> AcceptanceProbability:
        return self._acceptance

    @acceptance.setter
    def acceptance(self, strategy: AcceptanceProbability):
        self._acceptance = strategy

    @property
    def annotator(self) -> Optional[Annotator]:
        return self._annotator

    @annotator.setter
    def annotator(self, ann: Optional[Annotator]) -> None:
        if ann is not None and not isinstance(ann, Annotator):
            raise TypeError(
                f"annotator must be an Annotator instance. Got {type(ann)}.")
        self._annotator = ann
        if ann is not None:
            self._chain.enable_annotations()



    @property
    def subsamplers(self):
        """Flat list of nested inner samplers, outermost-inner first, coarsest root last.

        Empty for samplers whose proposal does not run an inner chain. Replaces
        manual traversal of `proposal.measure.mcmc`.
        """
        result = []
        proposal = self._proposalMethod
        while True:
            measure = getattr(proposal, "measure", None)
            if measure is None:
                break
            inner = getattr(measure, "mcmc", None)
            if inner is None:
                break
            result.append(inner)
            proposal = inner.proposal
        return result


    @abstractmethod
    def _log_mh_ratio(self, transition: TransitionData) -> float:
        """Compute log MH ratio from a proposal transition."""
        ...

    def _accept_reject(self, transition: TransitionData) -> TransitionData:
        logMHRatio = self._log_mh_ratio(transition)
        logAcceptProb = self._acceptance.log_probability(logMHRatio)

        outcome = (TransitionData.ACCEPTED if log(self._rng.uniform()) <= logAcceptProb
                   else TransitionData.REJECTED)
        return TransitionData(
            transition.state, transition.proposal, outcome, transition.auxiliary
        )

    def _update_chain(self, nextState):
        if not self._storeChain:
            return
        annotation = self._annotator.annotate(
            nextState) if self._annotator else None
        self._chain.append(nextState.coordinate, annotation)

    def _determine_next_state(self, transitionData):
        if transitionData.outcome == TransitionData.ACCEPTED:
            return transitionData.proposal

        if transitionData.outcome == TransitionData.REJECTED:
            return transitionData.state

        raise ValueError(
            f"Invalid transition outcome: {transitionData.outcome}")

    def _process_transition(self, transitionData):
        self._diagnostics.process(transitionData)
        nextState = self._determine_next_state(transitionData)
        self._update_chain(nextState)
        return nextState

    def _iterate(self) -> Parameter:
        """Perform a single Metropolis-Hastings transition."""

        self._proposalMethod.state = self._lastState.clone()
        transition = self._proposalMethod.generate_proposal(self._rng)

        transitionOutcome = self._accept_reject(transition)
        self._lastState = self._process_transition(transitionOutcome)

        return self._lastState

    def clear(self):
        """Reset diagnostics and discard all chain history."""
        super().clear()
        self._chain.clear()
        for sampler in self.subsamplers:
            sampler.diagnostics.reset()
