import numpy as np
import pytest

from styne.backend import get_backend
from styne.mcmc.chain import Chain
from styne.mcmc.diagnostics import DummyDiagnostics
from styne.mcmc.method.mrw import MetropolisedRandomWalk
from styne.mcmc.mixture import Mixture
from styne.mcmc.sampler import MCMCSampler
from styne.parameter import Vector
from styne.statistics import Gaussian, IIDCovarianceMatrix
from styne.utility import RobbinsMonroAdaptation


def setup_mixture(backendName="numpy", weights=None):
    backend = get_backend(backendName)
    start = Vector(backend.zeros(2, dtype="float32"))
    target = Gaussian(IIDCovarianceMatrix(2, backend.asarray(1.0)), start)
    components = [
        MetropolisedRandomWalk(
            target.density, IIDCovarianceMatrix(2, backend.asarray(scale)),
            DummyDiagnostics(),
            adaptation=RobbinsMonroAdaptation() if index else None,
        )
        for index, scale in enumerate((0.1, 1.0))
    ]
    mixture = Mixture(components, weights, rng=backend.random_state(2026))
    return backend, start, target, mixture


def array(value):
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)


@pytest.mark.parametrize("backendName", ["numpy", "jax", "pytorch"])
def test_outer_history_and_component_history(backendName):
    backend, start, _, mixture = setup_mixture(backendName, [0.0, 1.0])
    mixture.run(8, start)
    mixture.continue_run(3)
    assert mixture.chain.length == 12
    assert array(mixture.currentState.componentStates[1].adaptationState.stepCount) == 11
    assert mixture.currentState.componentStates[0].adaptationState is None
    assert all(component.chain.length == 0 for component in mixture.components)
    assert mixture.currentState.parameter is mixture.lastState


def test_validation_and_state_at_compatibility():
    _, start, _, mixture = setup_mixture()
    with pytest.raises(ValueError):
        Mixture([])
    with pytest.raises(ValueError):
        Mixture(mixture.components, [np.nan, 1.0])
    with pytest.raises(ValueError):
        Mixture(mixture.components, [1.0])
    with pytest.raises(ValueError):
        Mixture([mixture.components[0], setup_mixture()[3].components[1]])
    base = mixture.components[0]
    with pytest.raises(NotImplementedError):
        MCMCSampler.state_at(base, start, object())
    assert MCMCSampler.state_at(base, start, start) is start
    saved = mixture.components[1].initial_state(start)
    current = base.initial_state(start.with_coordinate(np.ones(2, dtype="float32")))
    moved = mixture.components[1].state_at(current, saved)
    assert moved.parameter is current.parameter
    assert moved.logDensity is current.logDensity
    assert moved.adaptationState is saved.adaptationState


def test_weights_do_not_alias_caller_arrays():
    _, _, _, original = setup_mixture()
    weights = np.array([1.0, 3.0])
    mixture = Mixture(original.components, weights)
    weights[:] = 0.0
    exposed = mixture.weights
    exposed[:] = 0.0
    np.testing.assert_array_equal(mixture.weights, [0.25, 0.75])


def test_nested_records_and_retargeting():
    _, start, target, inner = setup_mixture(weights=[0, 1])
    outer = Mixture([inner.components[0], inner], [0, 1])
    state, record, _ = outer.step(outer.initial_state(start), np.random.default_rng(9))
    assert record.component == 1
    assert record.records[1].component == 1
    assert record.records[1].records[0] is None
    assert state.componentStates[1].componentStates[1].adaptationState.stepCount == 1
    retargeted = outer.with_target(target.density)
    assert all(component.target is target.density for component in retargeted.components)
    assert (
        retargeted.components[0].proposal.covariance
        is outer.components[0].proposal.covariance
    )


def test_no_target_reevaluation_when_switching_components():
    _, start, target, mixture = setup_mixture()
    calls = []
    evaluate = target.density.evaluate_log
    target.density.evaluate_log = lambda value: (calls.append(1), evaluate(value))[1]
    state = mixture.initial_state(start)
    calls.clear()
    rng = np.random.default_rng(4)
    for _ in range(12):
        state, _, rng = mixture.step(state, rng)
    assert len(calls) == 12


class IndependentSampler(MCMCSampler):
    def __init__(self, target):
        super().__init__()
        self._measure = target
        self._chain = Chain()

    @property
    def target(self):
        return self._measure.density

    @property
    def chain(self):
        return self._chain

    def step(self, state, rng):
        draw, rng = self._measure.sample(rng)
        return draw, None, rng

    def zero_sums(self, state):
        return state.backend.asarray(0.0)

    def add_to_sums(self, sums, transition):
        return sums + 1.0


def test_non_mh_component():
    _, start, target, mixture = setup_mixture()
    mixed = Mixture([mixture.components[0], IndependentSampler(target)], [0, 1])
    state, record, _ = mixed.step(mixed.initial_state(start), np.random.default_rng(2))
    assert record.records == (None, None)
    np.testing.assert_allclose(
        state.logDensity, target.density.evaluate_log(state.parameter)
    )


def test_jax_jit_scan_resume_and_private_history():
    jax = pytest.importorskip("jax")
    backend, start, _, mixture = setup_mixture("jax", [0.4, 0.6])
    key = backend.random_state(17)
    run = mixture.compiled_trajectory(20)
    whole = run(start, key)
    halfRun = mixture.compiled_trajectory(10)
    half = halfRun(start, key)
    resumed = halfRun(half.finalState, half.rng)
    np.testing.assert_allclose(whole.final.coordinate, resumed.final.coordinate)
    np.testing.assert_array_equal(
        jax.random.key_data(whole.rng), jax.random.key_data(resumed.rng)
    )
    selected = whole.sums["selected"]
    assert np.sum(selected) == 20
    assert whole.finalState.componentStates[1].adaptationState.stepCount == selected[1]
    assert whole.coordinates.shape == (20, 2)


def test_jax_nested_heterogeneous_records():
    jax = pytest.importorskip("jax")
    backend, start, target, inner = setup_mixture("jax")
    outer = Mixture([inner, IndependentSampler(target)])
    result = outer.compiled_trajectory(12)(start, backend.random_state(19))
    assert result.coordinates.shape == (12, 2)
    assert np.sum(result.sums["selected"]) == 12
    assert np.sum(result.sums["components"][0]["selected"]) == result.sums["selected"][0]


def test_jax_batched_choices_and_resume():
    pytest.importorskip("jax")
    backend, _, _, mixture = setup_mixture("jax", [0.4, 0.6])
    start = Vector(backend.zeros((4, 2), dtype="float32"))
    key = backend.random_state(18)
    whole = mixture.compiled_trajectory(24)(start, key)
    halfRun = mixture.compiled_trajectory(12)
    half = halfRun(start, key)
    resumed = halfRun(half.finalState, half.rng)
    np.testing.assert_allclose(whole.final.coordinate, resumed.final.coordinate)
    assert whole.coordinates.shape == (24, 4, 2)
    selected = np.asarray(whole.sums["selected"])
    np.testing.assert_array_equal(selected.sum(axis=1), np.full(4, 24))
    assert np.unique(selected[:, 0]).size > 1
    np.testing.assert_array_equal(
        whole.finalState.componentStates[1].adaptationState.stepCount, selected[:, 1]
    )


def test_jax_batch_runs_only_selected_components():
    jax = pytest.importorskip("jax")
    backend, _, target, _ = setup_mixture("jax")
    calls = [[], []]

    class CountedWalk(MetropolisedRandomWalk):
        def __init__(self, index):
            super().__init__(
                target.density, IIDCovarianceMatrix(2, backend.asarray(0.2)),
                DummyDiagnostics(),
            )
            self._index = index

        def step(self, state, rng):
            jax.debug.callback(lambda: calls[self._index].append(1), ordered=True)
            return super().step(state, rng)

    mixture = Mixture([CountedWalk(0), CountedWalk(1)], [0.4, 0.6])
    start = Vector(backend.zeros((4, 2), dtype="float32"))
    result = mixture.compiled_trajectory(6)(start, backend.random_state(28))
    result.coordinates.block_until_ready()
    jax.effects_barrier()
    np.testing.assert_array_equal(
        [len(count) for count in calls], np.asarray(result.sums["selected"]).sum(axis=0)
    )
    assert sum(map(len, calls)) == 24


def test_jax_batch_non_mh_component_sums():
    pytest.importorskip("jax")
    backend, _, target, base = setup_mixture("jax")
    mixture = Mixture([base.components[0], IndependentSampler(target)])
    start = Vector(backend.zeros((3, 2), dtype="float32"))
    result = mixture.compiled_trajectory(10)(start, backend.random_state(36))
    np.testing.assert_array_equal(
        result.sums["components"][1], result.sums["selected"][:, 1]
    )


@pytest.mark.parametrize("backendName", ["numpy", "jax"])
def test_dart_factory_with_mixture_components_and_mixture_root(backendName):
    from styne.mcmc.method.dart import DARTFactory

    backend, start, target, base = setup_mixture(backendName)
    first = base.components[0]
    second = MetropolisedRandomWalk(
        target.density, IIDCovarianceMatrix(2, backend.asarray(0.6)),
        DummyDiagnostics(),
    )
    root = Mixture([first, second], [0.5, 0.5])
    factory = DARTFactory(root=root)
    factory.target = target.density
    factory.surrogate = [target.density]
    factory.rootLength = [4]
    factory.correctionType = "importance"
    factory.components = [{"penaltyMetric": 0.3}, {"penaltyMetric": 2.0}]
    factory.weights = [0.4, 0.6]
    sampler = factory.create()
    assert isinstance(sampler, Mixture)
    assert factory.localisedDensity is None
    assert all(component.target is target.density for component in sampler.components)
    assert root.target is target.density
    key = backend.random_state(37)
    if backendName == "jax":
        result = sampler.compiled_trajectory(3)(start, key)
        state = result.finalState
        assert np.sum(result.sums["selected"]) == 3
    else:
        state = sampler.initial_state(start)
        for _ in range(3):
            state, _, key = sampler.step(state, key)
    np.testing.assert_allclose(
        state.logDensity, target.density.evaluate_log(state.parameter), rtol=1e-5
    )
