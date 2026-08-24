from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Optional
from numpy.random import Generator, default_rng
from tqdm.contrib.logging import logging_redirect_tqdm

from styne.utility.progress import make_reporter

from styne.parameter.parameter import Parameter
from styne.mcmc.chain import Chain
from styne.mcmc.diagnostics import DummyDiagnostics


class MCMCSampler(ABC):
    """Abstract base class for MCMC samplers.

    Subclasses implement '_iterate', which defines the sampling procedure for one
    iteration. The public interface is 'run', which repeatedly calls '_iterate'
    to generate samples and manages shared bookkeeping like the chain and the
    last produced state.
    """

    def __init__(self, rng: Optional[Generator] = None):
        self._lastState: Parameter = None
        self._iteration: int = 0
        self._storeChain: bool = True
        self._rng = rng if rng is not None else default_rng()
        self._diagnostics = DummyDiagnostics()
        


    @property
    def storeChain(self) -> bool:
        return self._storeChain

    @storeChain.setter
    def storeChain(self, value: bool):
        self._storeChain = value

    @property
    def diagnostics(self):
        return self._diagnostics

    @diagnostics.setter
    def diagnostics(self, value):
        self._diagnostics = value

    @property
    def lastState(self) -> Parameter:
        """Most recently produced state."""
        return self._lastState

    @property
    @abstractmethod
    def chain(self) -> Chain:
        """The chain of samples produced by this sampler."""
        ...

    @abstractmethod
    def _iterate(self) -> Parameter:
        """Perform one sampling iteration and return the next state."""
        ...

    def _initialize(
            self, nSteps: int, initialState: Parameter):
        """Optional hook for subclasses to prepare before a run."""
        pass

    def clear(self):
        """Reset sampler state (default: clear lastState)."""
        self._lastState = None
        self._iteration = 0
        self._diagnostics.clear()

    def _validate_initial(self, initialState):
        if not isinstance(initialState, Parameter):
            raise ValueError(
                "Initial mcmc chain state must adhere to Parameter interface. "
                f"Got type {type(initialState)} instead."
            )

    def _append_initial(self, initialState):
        if self._storeChain:
            self.chain.append(initialState.coordinate)

    def _drive(self, nSteps, progress, description):
        samplerName = getattr(self, "name", self.__class__.__name__)
        description = description or f"[{samplerName}]"
        redirect = logging_redirect_tqdm() if progress else nullcontext()
        reporterCtx = make_reporter(
            progress, total=nSteps, description=description
        )
        with redirect, reporterCtx as reporter:
            for n in range(nSteps):
                self._iteration = n
                self._lastState = self._iterate()
                reporter.update(1, **self._diagnostics.summary())
                yield self._lastState

    def stream_run(self, nSteps, initialState, progress=False, description=None):
        """Generator form of 'run'. Clears, then yields the state after each transition.

        Use when interleaving per-step work such as accumulating a running prediction
        or stopping early.
        """
        self._validate_initial(initialState)
        self.clear()
        self._initialize(nSteps, initialState)
        self._append_initial(initialState)
        self._lastState = initialState
        yield from self._drive(nSteps, progress, description)

    def run(self, nSteps, initialState, progress=False, description=None):
        """Run a fresh chain for a fixed number of steps.

        Each call clears sampler state first, last state, iteration counter,
        and diagnostics, then starts from 'initialState'. Concrete samplers
        also discard prior chain history as part of their own 'clear'. Use
        this for a long single run. To run, pause, act, and resume without
        clearing, use 'continue_run'.
        """
        for _ in self.stream_run(nSteps, initialState, progress, description):
            pass

    def stream_continue(self, nSteps, progress=False, description=None):
        """Generator form of 'continue_run'. Resumes from 'lastState' without clearing."""
        if self._lastState is None:
            raise RuntimeError("Cannot continue: chain is empty. Call 'run' first.")
        self._initialize(nSteps, self._lastState)
        yield from self._drive(nSteps, progress, description)

    def continue_run(self, nSteps, progress=False, description=None):
        """Continue the chain from the last recorded state without clearing.

        Preserves cumulative diagnostics and the existing chain. Requires a prior
        'run'. This is the lightweight resume counterpart to 'run'.
        """
        for _ in self.stream_continue(nSteps, progress, description):
            pass
