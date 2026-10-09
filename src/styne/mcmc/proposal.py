from styne.statistics.measure import (
    AbsolutelyContinuousProbabilityMeasure,
    ProbabilityMeasure,
)
from abc import ABC, abstractmethod
from typing import List, Optional
from numpy.random import Generator

from styne.mcmc.transition import TransitionData
from styne.parameter.parameter import Parameter
from styne.parameter.vector import Vector
from styne.utility.partition import Partition


def log_reference_ratio(reference, current: Parameter, proposed: Parameter):
    """Log density ratio of a reference measure, zero for the Lebesgue measure."""
    if reference is None:
        return 0.0
    return (
        reference.density.evaluate_log(proposed)
        - reference.density.evaluate_log(current)
    )


def zero_correction(state: Parameter):
    """Zero log correction with the batch shape of ``state``."""
    metadata = state.backendMetadata
    return state.backend.zeros(
        state.coordinate.shape[:-1],
        dtype=metadata.dtype,
        device=metadata.device,
    )


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
    Subclasses implement :meth:`propose`, which returns the proposed state
    and its log acceptance correction log q_y(x) - log q_x(y), for proposal
    densities q relative to :attr:`reference`, under the auxiliary key
    ``logCorrection``. Every numerical input is explicit; proposal instances
    retain only static configuration.
    """

    @property
    def reference(self) -> Optional[ProbabilityMeasure]:
        """Measure the acceptance correction refers to, None for Lebesgue."""
        return None

    def condition(self, state: Parameter) -> "ProposalMethod":
        """
        Return this proposal given the current state ``state``.

        A sampler conditions its proposal once before each transition. The
        conditioned proposal may depend on the state only through the
        coordinates it holds, so that the forward and the reverse move share
        it. A proposal independent of the state returns itself, the default.
        """
        return self

    def for_target(self, target, previous) -> "ProposalMethod":
        """
        Return this proposal for ``target`` in place of ``previous``.

        A sampler calls this when its target changes. Whatever the proposal
        took from ``previous``, such as its reference or gradient, it takes
        from ``target`` instead, and anything given explicitly is kept. A
        proposal that took nothing returns itself, the default.
        """
        return self

    @staticmethod
    def record(current: Parameter, proposed: Parameter, logCorrection,
               **auxiliary):
        """Proposal record carrying the log acceptance correction."""
        return TransitionData(
            current=current,
            proposed=proposed,
            auxiliary={"logCorrection": logCorrection, **auxiliary},
        )

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
        if len(proposalMethods) != partition.rule.numComponents:
            raise ValueError("One proposal is required per partition component.")
        self._partition = partition
        self._pMethods = list(proposalMethods)

    @property
    def components(self) -> List[ProposalMethod]:
        """The constituent proposal methods for each partition component."""
        return list(self._pMethods)

    def condition(self, state: Parameter):
        proposals = [method.condition(state) for method in self._pMethods]
        return type(self)(self._partition, proposals)

    def for_target(self, target, previous):
        return type(self)(self._partition, [
            method.for_target(target, previous) for method in self._pMethods
        ])

    def propose(self, state: Parameter, rng: Generator):
        coordinates = []
        correction = zero_correction(state)
        for index, method in enumerate(self._pMethods):
            current = Vector(self._partition.rule.extract(index, state.coordinate))
            transition, rng = method.propose(current, rng)
            proposed = transition.proposed.parameter
            correction = (
                correction + transition.auxiliary["logCorrection"]
                - log_reference_ratio(method.reference, current, proposed)
            )
            coordinates.append(proposed.coordinate)
        proposed = state.with_coordinate(self._partition.rule.merge(coordinates))
        return self.record(state, proposed, correction), rng


class FixedProposal(ProposalMethod):
    """
    Independence proposal from a fixed measure.

    Its draws are reversible with respect to the proposal measure, so the
    correction is zero relative to that measure.
    """

    def __init__(self, proposalMeasure: AbsolutelyContinuousProbabilityMeasure):

        if not isinstance(
                proposalMeasure, AbsolutelyContinuousProbabilityMeasure):
            raise TypeError(
                "proposalMeasure must be an "
                "AbsolutelyContinuousProbabilityMeasure instance."
            )

        self._proposal = proposalMeasure

    @property
    def reference(self) -> AbsolutelyContinuousProbabilityMeasure:
        return self._proposal

    def propose(self, state, rng):
        proposal, nextRng = self._proposal.sample(rng)
        return self.record(state, proposal, zero_correction(state)), nextRng


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
        coarse, fine = self._coarse_from(state), self._fine_from(state)
        coarseTransition, rng = self._coarseProposal.propose(coarse, rng)
        fineTransition, nextRng = self._fineKernel.propose(fine, rng)
        coarseProposal = coarseTransition.proposed.parameter
        fineProposal = fineTransition.proposed.parameter
        # The blocks refer to different measures, so both corrections are
        # taken relative to the Lebesgue measure.
        logCorrection = (
            coarseTransition.auxiliary["logCorrection"]
            - log_reference_ratio(
                self._coarseProposal.reference, coarse, coarseProposal
            )
            + fineTransition.auxiliary["logCorrection"]
            - log_reference_ratio(self._fineKernel.reference, fine, fineProposal)
        )
        fullProposal = self._merge(
            coarseProposal.coordinate, fineProposal.coordinate, state
        )
        return self.record(state, fullProposal, logCorrection), nextRng
