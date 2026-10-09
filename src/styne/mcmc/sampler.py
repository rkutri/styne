from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, NamedTuple, Optional
from numpy.random import Generator, default_rng
from tqdm.contrib.logging import logging_redirect_tqdm

from styne.utility.progress import make_reporter

from styne.backend import infer_backend
from styne.parameter.parameter import Parameter
from styne.mcmc.chain import Chain
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.transition import parameter_of


class Trajectory(NamedTuple):
    """Result of a compiled trajectory.

    'coordinates' holds the coordinates after every 'thinning'-th transition,
    'readouts' the caller's read-out after every transition or None, and
    'sums' the running sums of the transition records.
    """

    final: Parameter
    coordinates: Any
    readouts: Any
    sums: Any
    rng: Any


class MCMCSampler(ABC):
    """Abstract base class for MCMC samplers.

    Subclasses implement '_iterate', which defines the sampling procedure for one
    iteration. The public interface is 'run', which repeatedly calls '_iterate'
    to generate samples and manages shared bookkeeping like the chain and the
    last produced state.
    """

    def __init__(self, rng: Optional[Generator] = None):
        self._lastState: Parameter = None
        self._runnerState = None
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

    def _iterate(self) -> Parameter:
        """Legacy stateful iteration hook for samplers not yet ported."""
        raise NotImplementedError

    def initial_state(self, parameter: Parameter):
        """Construct pure-transition state from an initial parameter."""
        return parameter

    def step(self, currentState, rng):
        """Return next numerical state, transition data, and random state."""
        raise NotImplementedError

    def _initialize(
            self, nSteps: int, initialState: Parameter):
        """Optional hook for subclasses to prepare before a run."""
        pass

    def clear(self):
        """Reset sampler state (default: clear lastState)."""
        self._lastState = None
        self._runnerState = None
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

    def _uses_pure_step(self):
        return type(self).step is not MCMCSampler.step

    @staticmethod
    def _parameter_from_state(state):
        return getattr(state, "parameter", state)

    def _record_transition(self, transitionData, nextState):
        self._diagnostics.process(transitionData)
        if self._storeChain:
            parameter = self._parameter_from_state(nextState)
            self.chain.append(parameter.coordinate)

    def zero_sums(self, state):
        """Running sums of no transition, for a trajectory from ``state``."""
        return {}

    def add_to_sums(self, sums, transition):
        """``sums`` with the record of one more transition added."""
        return sums

    def transformed_trajectory(self, nSteps, initialState, rng=None,
                               thinning=1, readout=None) -> Trajectory:
        """Run pure transitions as one backend-native computation.

        The loop carries the running sums of the transition records, so that
        no record outlives its transition. The coordinates are kept after
        every 'thinning'-th transition, which must divide 'nSteps', and
        'readout' maps the parameter after every transition to an array.
        Each call compiles anew. For a run in chunks, see
        'compiled_trajectory'.
        """
        return self.compiled_trajectory(nSteps, thinning, readout)(
            initialState, rng
        )

    def compiled_trajectory(self, nSteps, thinning=1, readout=None):
        """Return 'transformed_trajectory' as a function compiled once.

        The function maps an initial parameter and random state to a
        Trajectory. Compilation is reused for chunks with compatible input
        shapes and dtypes. Keep the sampler configuration fixed while using
        this function; changing its target or proposal requires a new one.
        """
        if not self._uses_pure_step():
            raise RuntimeError("Transformed trajectories require a sampler with step().")
        if isinstance(nSteps, bool) or not isinstance(nSteps, int) or nSteps < 1:
            raise ValueError("Transformed trajectories require positive integer nSteps.")
        if isinstance(thinning, bool) or not isinstance(thinning, int) \
                or thinning < 1 or nSteps % thinning:
            raise ValueError(
                f"thinning must divide nSteps. Got {thinning} and {nSteps}."
            )
        compiled = {}

        def run(initialState, rng=None) -> Trajectory:
            self._validate_initial(initialState)
            backend = infer_backend(initialState.coordinate)
            if not backend.capabilities.transformedLoops:
                raise RuntimeError(
                    f"Backend {backend.name!r} does not support transformed loops."
                )
            if backend.name not in compiled:
                compiled[backend.name] = backend.compile(
                    self._trajectory_loop(backend, nSteps, thinning, readout)
                )
            randomState = self._rng if rng is None else rng
            (finalState, nextRng, sums), (coordinates, readouts) = \
                compiled[backend.name](initialState, randomState)
            if readouts is not None:
                readouts = readouts.reshape((nSteps,) + tuple(readouts.shape[2:]))
            return Trajectory(
                parameter_of(finalState), coordinates, readouts, sums, nextRng
            )
        return run

    def _trajectory_loop(self, backend, nSteps, thinning, readout):
        def advance(carry, _):
            state, currentRng, sums = carry
            nextState, transition, nextRng = self.step(state, currentRng)
            output = None if readout is None else readout(parameter_of(nextState))
            return (nextState, nextRng, self.add_to_sums(sums, transition)), output

        # An inner loop over the thinning interval keeps one coordinate each.
        def advance_block(carry, _):
            carry, outputs = backend.scan(advance, carry, None, length=thinning)
            return carry, (parameter_of(carry[0]).coordinate, outputs)

        def execute(parameter, currentRng):
            state = self.initial_state(parameter)
            return backend.scan(
                advance_block, (state, currentRng, self.zero_sums(state)), None,
                length=nSteps // thinning,
            )
        return execute

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
                if self._uses_pure_step():
                    nextState, transition, self._rng = self.step(
                        self._runnerState, self._rng
                    )
                    self._runnerState = nextState
                    self._record_transition(transition, nextState)
                    self._lastState = self._parameter_from_state(nextState)
                else:
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
        if self._uses_pure_step():
            self._runnerState = self.initial_state(initialState)
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
        if self._uses_pure_step():
            if self._runnerState is None:
                self._runnerState = self.initial_state(self._lastState)
        else:
            self._initialize(nSteps, self._lastState)
        yield from self._drive(nSteps, progress, description)

    def continue_run(self, nSteps, progress=False, description=None):
        """Continue the chain from the last recorded state without clearing.

        Preserves cumulative diagnostics and the existing chain. Requires a prior
        'run'. This is the lightweight resume counterpart to 'run'.
        """
        for _ in self.stream_continue(nSteps, progress, description):
            pass
