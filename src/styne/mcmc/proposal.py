from styne.statistics.measure import ProbabilityMeasure
from abc import ABC, abstractmethod
from typing import List, Optional
from numpy.random import Generator

from styne.mcmc.transition import TransitionData
from styne.parameter.parameter import Parameter
from styne.parameter.vector import Vector
from styne.utility.partition import Partition


class PartitionedProposalMixin:
    """Mixin for proposals that split a full-dimensional step into a
    coarse surrogate transition and an independent fine prior draw.

    Call `_init_partition` from the subclass constructor when a
    partition is provided.  All methods are no-ops / raise when used
    without a partition, so the mixin is safe to inherit unconditionally.
    """

    def _init_partition(
        self,
        partition: Optional[Partition],
        finePrior: Optional[ProbabilityMeasure],
    ) -> None:
        self._pPartition = partition
        self._pFinePrior = finePrior

    @property
    def is_partitioned(self) -> bool:
        return getattr(self, '_pPartition', None) is not None

    def _coarse_from(self, state: Parameter) -> Vector:
        return Vector(self._pPartition.rule.extract(0, state.coordinate))

    def _fine_from(self, state: Parameter) -> Vector:
        return Vector(self._pPartition.rule.extract(1, state.coordinate))

    def _draw_fine(self, rng: Generator) -> Vector:
        return self._pFinePrior.generate_realisation(rng=rng)

    def _merge(self, coarseCoord, fineCoord, template: Parameter) -> Parameter:
        result = template.clone()
        result.coordinate = self._pPartition.rule.merge(
            [coarseCoord, fineCoord])
        return result


class ProposalMethod(ABC):

    def __init__(self):
        self._state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @abstractmethod
    def generate_proposal(self, rng: Generator) -> TransitionData:
        ...


class BlockProposal(ProposalMethod):
    """
    Proposal based on a partition of the parameter.

    Each partition component is proposed independently by its own
    ProposalMethod. The full proposal is assembled by merging the
    component proposals via the partition.
    """

    def __init__(self, partition: Partition, proposalMethods: List[ProposalMethod]):
        """
        Parameters
        ----------
        partition : Partition
            Partition of the full parameter coordinate.
        proposalMethods : List[ProposalMethod]
            One proposal method per partition component, in the same order
            as the partition rule.
        """
        super().__init__()
        self._partition = partition
        self._pMethods = proposalMethods

    @property
    def components(self) -> List[ProposalMethod]:
        """The constituent proposal methods for each partition component."""
        return list(self._pMethods)

    @ProposalMethod.state.setter
    def state(self, state: Parameter):

        ProposalMethod.state.fset(self, state)

        for idx, method in enumerate(self._pMethods):
            method.state = Vector(self._partition.component(idx, state))

    def generate_proposal(self, rng: Generator) -> TransitionData:

        components = [
            m.generate_proposal(rng).proposal.coordinate for m in self._pMethods
        ]

        result = self._state.clone()
        result.coordinate = self._partition.merge(components)

        return TransitionData(self._state, result)


class FixedProposal(ProposalMethod):
    """
    State-independent proposal.
    """

    def __init__(self, proposalMeasure: ProbabilityMeasure):

        if not isinstance(proposalMeasure, ProbabilityMeasure):
            raise TypeError(
                "proposalMeasure must be a ProbabilityMeasure instance."
            )

        self._proposal = proposalMeasure

    @property
    def state(self):
        raise RuntimeError("FixedProposal has no state.")

    # state setter is a no-op
    @ProposalMethod.state.setter
    def state(self, state):
        pass

    def generate_proposal(self, rng: Generator) -> TransitionData:
        return TransitionData(None, self._proposal.generate_realisation(rng=rng))


class PartitionedSurrogateProposal(PartitionedProposalMixin, ProposalMethod):
    """Full-dimensional proposal splitting into a coarse surrogate step and a fine
    kernel draw. The fine kernel must preserve 'finePrior', so its only contribution
    to the acceptance ratio is the fine-prior difference.
    """

    def __init__(self, partition, coarseProposal, fineKernel, finePrior):
        ProposalMethod.__init__(self)
        self._init_partition(partition, finePrior)
        self._coarseProposal = coarseProposal
        self._fineKernel = fineKernel

    @property
    def measure(self):
        return self._coarseProposal.measure

    @property
    def density(self):
        return self._coarseProposal.density

    @ProposalMethod.state.setter
    def state(self, state):
        ProposalMethod.state.fset(self, state)
        self._coarseProposal.state = self._coarse_from(state)
        self._fineKernel.state = self._fine_from(state)

    def generate_proposal(self, rng):
        coarseProposal = self._coarseProposal.generate_proposal(rng).proposal
        fineProposal = self._fineKernel.generate_proposal(rng).proposal
        fullProposal = self._merge(
            coarseProposal.coordinate, fineProposal.coordinate, self._state)
        return TransitionData(self._state, fullProposal)

    def log_acceptance_correction(self, state, proposal):
        coarseState = self._coarse_from(state)
        coarseProposal = self._coarse_from(proposal)
        fineState = self._fine_from(state)
        fineProposal = self._fine_from(proposal)
        finePriorDensity = self._pFinePrior.density
        fineCorrection = -(finePriorDensity.evaluate_log(fineProposal)
                           - finePriorDensity.evaluate_log(fineState))
        coarseCorrection = self._coarseProposal.log_acceptance_correction(
            coarseState, coarseProposal)
        return coarseCorrection + fineCorrection