"""Strategies for updating proposal settings from completed MH transitions."""
from abc import ABC, abstractmethod

from styne.mcmc.proposal import ProposalMethod


class Adaptation(ABC):
    """Static adaptation rule; evolving information belongs in its state.

    States must be immutable trees of backend arrays with fixed structure,
    shapes and dtypes during compiled execution. Methods must not mutate the
    strategy, proposal template or input state. Updating need not stop.
    """

    @abstractmethod
    def initial_state(self, parameter, proposalMethod):
        """Initialise numerical history and validate the proposal family."""
        ...

    @abstractmethod
    def prepare_proposal(self, proposalMethod, state) -> ProposalMethod:
        """Return the proposal configured for one complete MH transition."""
        ...

    @abstractmethod
    def update_state(self, state, nextState, transitionData):
        """Return history for the next transition, including rejected-state repeats.

        The transition supplies both evaluated endpoints, the acceptance outcome,
        log acceptance probability and auxiliary proposal data. Return the state
        unchanged when no update is due. No runner diagnostics are needed.
        """
        ...


class NoAdaptation(Adaptation):
    """Keep the configured proposal and carry no adaptation history."""

    def initial_state(self, parameter, proposalMethod):
        return None

    def prepare_proposal(self, proposalMethod, state):
        return proposalMethod

    def update_state(self, state, nextState, transitionData):
        return None
