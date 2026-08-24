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
    def isPartitioned(self) -> bool:
        return getattr(self, '_pPartition', None) is not None

    def _coarse_from(self, state: Parameter) -> Vector:
        return Vector(self._pPartition.rule.extract(0, state.coordinate))

    def _fine_from(self, state: Parameter) -> Vector:
        return Vector(self._pPartition.rule.extract(1, state.coordinate))

    def _draw_fine(self, rng: Generator) -> Vector:
        return self._pFinePrior.generate_realisation(rng=rng)

    def _merge(self, coarseCoord, fineCoord, template: Parameter) -> Parameter:
        coordinate = self._pPartition.rule.merge([coarseCoord, fineCoord])
        return template.with_coordinate(coordinate)


class ProposalMethod(ABC):
    """
    Interface for MCMC proposal mechanisms.

    Notes
    -----
    Subclasses implement :meth:`propose`. Every numerical input is explicit;
    proposal instances retain only static configuration.
    """

    @abstractmethod
    def propose(
            self, state: Parameter, rng: Generator
    ) -> tuple[TransitionData, object]:
        """
        Generate a proposal from ``state`` using ``rng``.

        Parameters
        ----------
        state : Parameter
            Current chain state.
        rng : object
            Backend-native random state.

        Returns
        -------
        (TransitionData, object)
            Proposal record and propagated random state.
        """
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
        self._partition = partition
        self._pMethods = proposalMethods

    @property
    def components(self) -> List[ProposalMethod]:
        """The constituent proposal methods for each partition component."""
        return list(self._pMethods)

    def propose(self, state: Parameter, rng: Generator):
        """
        Generate a proposal by drawing independently from each component
        proposal method and merging the results via the partition.

        Parameters
        ----------
        rng : Generator

        Returns
        -------
        TransitionData
        """

        components = [
            method.propose(
                Vector(self._partition.component(index, state)), rng
            )
            for index, method in enumerate(self._pMethods)
        ]
        transitions, nextRngs = zip(*components)
        result = state.with_coordinate(
            self._partition.merge([
                transition.proposal.coordinate for transition in transitions
            ])
        )
        return TransitionData(state, result), nextRngs[-1]


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

    def propose(self, state, rng):
        proposal, nextRng = self._proposal.sample(rng)
        return TransitionData(state, proposal), nextRng


class PartitionedSurrogateProposal(PartitionedProposalMixin, ProposalMethod):
    """Full-dimensional proposal splitting into a coarse surrogate step and a fine
    kernel draw. The fine kernel must preserve 'finePrior', so its only contribution
    to the acceptance ratio is the fine-prior difference.
    """

    def __init__(self, partition, coarseProposal, fineKernel, finePrior):
        self._init_partition(partition, finePrior)
        self._coarseProposal = coarseProposal
        self._fineKernel = fineKernel

    @property
    def measure(self):
        return self._coarseProposal.measure

    @property
    def density(self):
        return self._coarseProposal.density

    def propose(self, state, rng):
        coarseTransition, rng = self._coarseProposal.propose(
            self._coarse_from(state), rng
        )
        fineTransition, nextRng = self._fineKernel.propose(
            self._fine_from(state), rng
        )
        fullProposal = self._merge(
            coarseTransition.proposal.coordinate,
            fineTransition.proposal.coordinate,
            state,
        )
        return TransitionData(state, fullProposal), nextRng

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
