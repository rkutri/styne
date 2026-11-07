"""Fixed mixtures of complete MCMC transitions."""
import copy
from typing import NamedTuple

from styne.backend import infer_backend
from styne.mcmc.chain import Chain
from styne.mcmc.sampler import MCMCSampler
from styne.mcmc.transition import EvaluatedState, parameter_of, stored_log_density
from styne.statistics.mixture import Mixture as Components


class MixtureState(NamedTuple):
    current: EvaluatedState
    componentStates: tuple

    @property
    def parameter(self):
        return self.current.parameter

    @property
    def logDensity(self):
        return self.current.logDensity


class MixtureTransition(NamedTuple):
    """Selected component and its record; inactive slots must be ignored."""
    component: object
    records: tuple


class Mixture(MCMCSampler):
    """Select one complete transition with fixed, state-independent weights.

    Components share one target object and implement ``state_at``. Their
    private histories advance only when selected; this sampler owns the outer
    trajectory. A leading batch axis uses independent component draws per chain
    on JAX. NumPy and PyTorch support single-chain mixtures.

    Eager inactive record slots are None. Compiled dispatch fills them with
    numerical placeholders of fixed shape; only the selected slot is an event.
    """
    name = "Mixture"

    def __init__(self, components, weights=None, rng=None):
        super().__init__(rng=rng)
        self._mixture = Components(components, weights)
        self._components = self._mixture.components
        for component in self._components:
            if not isinstance(component, MCMCSampler) or not component._uses_pure_step():
                raise TypeError(
                    "Mixture components must implement pure MCMC transitions."
                )
        self._target = getattr(self._components[0], "target", None)
        if self._target is None or any(
                getattr(component, "target", None) is not self._target
                for component in self._components):
            raise ValueError("Mixture components must share one target object.")
        self._chain = Chain()

    @property
    def components(self):
        return self._components

    @property
    def weights(self):
        return self._mixture.rule.weights

    @property
    def target(self):
        return self._target

    @property
    def chain(self):
        if not self._storeChain:
            raise RuntimeError("Chain storage is disabled (storeChain=False).")
        return self._chain

    @property
    def subsamplers(self):
        return [
            sampler for component in self._components
            for sampler in (component, *getattr(component, "subsamplers", ()))
        ]

    def with_target(self, targetDensity, rng=None):
        result = type(self)(
            [component.with_target(targetDensity) for component in self._components],
            self.weights, copy.deepcopy(self._rng) if rng is None else rng,
        )
        result._diagnostics = copy.deepcopy(self._diagnostics)
        result._storeChain = self._storeChain
        result.clear()
        return result

    def clear(self):
        super().clear()
        self._chain.clear()

    def _validate_initial(self, parameter):
        super()._validate_initial(parameter)
        if parameter.coordinate.ndim > 2:
            raise ValueError("Mixture accepts one leading chain axis.")
        if parameter.coordinate.ndim == 2:
            if parameter.coordinate.shape[0] == 0:
                raise ValueError("A mixture batch needs at least one chain.")
            parameter = parameter.with_coordinate(parameter.coordinate[0])
        for component in self._components:
            component._validate_initial(parameter)

    def _shared_state(self, state):
        return EvaluatedState(
            parameter_of(state), stored_log_density(self.target, state)
        )

    def initial_state(self, parameter):
        if parameter.coordinate.ndim == 2:
            backend = infer_backend(parameter.coordinate)
            return backend.vectorize(self.initial_state)(parameter)
        states = tuple(
            component.initial_state(parameter) for component in self._components
        )
        return MixtureState(self._shared_state(states[0]), states)

    def state_at(self, current, saved):
        return MixtureState(self._shared_state(current), saved.componentStates)

    def _branch(self, index, emptyRecords):
        component = self._components[index]

        def advance(state, rng):
            local = component.state_at(state.current, state.componentStates[index])
            result, record, rng = component.step(local, rng)
            states = (
                state.componentStates[:index] + (result,)
                + state.componentStates[index + 1:]
            )
            records = emptyRecords[:index] + (record,) + emptyRecords[index + 1:]
            return MixtureState(self._shared_state(result), states), records, rng
        return advance

    def step(self, state, rng):
        backend = infer_backend(state.parameter.coordinate)
        if state.parameter.coordinate.ndim == 2:
            keys, rng = backend.split_random_state(
                rng, state.parameter.coordinate.shape[0]
            )
            nextState, transition, _ = backend.map(
                lambda operands: self.step(*operands), (state, keys)
            )
            return nextState, transition, rng
        index, rng = self._mixture.rule.draw(rng, state.parameter.coordinate)
        emptyRecords = (None,) * len(self._components)
        if backend.is_tracing(state.parameter.coordinate) or backend.is_tracing(index):
            emptyRecords = tuple(
                backend.zeros_like_output(
                    lambda local, key: component.step(local, key)[1],
                    component.state_at(state.current, saved), rng,
                )
                for component, saved in zip(self._components, state.componentStates)
            )
        nextState, records, rng = backend.switch(
            index,
            [self._branch(i, emptyRecords) for i in range(len(self._components))],
            state, rng,
        )
        return nextState, MixtureTransition(index, records), rng

    def zero_sums(self, state):
        coordinate = state.parameter.coordinate
        backend = infer_backend(coordinate)
        if coordinate.ndim == 2:
            return backend.vectorize(self.zero_sums)(state)
        metadata = backend.metadata(coordinate)
        return {
            "selected": backend.zeros(
                coordinate.shape[:-1] + (len(self._components),),
                dtype=metadata.dtype, device=metadata.device,
            ),
            "components": tuple(
                component.zero_sums(saved)
                for component, saved in zip(self._components, state.componentStates)
            ),
        }

    def add_to_sums(self, sums, transition):
        backend = infer_backend(sums["selected"])
        if sums["selected"].ndim == 2:
            return backend.map(lambda pair: self.add_to_sums(*pair), (sums, transition))

        def branch(index):
            def update():
                previous = sums["components"]
                value = self._components[index].add_to_sums(
                    previous[index], transition.records[index]
                )
                return previous[:index] + (value,) + previous[index + 1:]
            return update

        metadata = backend.metadata(sums["selected"])
        indices = backend.arange(
            len(self._components), dtype=metadata.dtype, device=metadata.device
        )
        return {
            "selected": sums["selected"] + (indices == transition.component),
            "components": backend.switch(
                transition.component,
                [branch(i) for i in range(len(self._components))],
            ),
        }
