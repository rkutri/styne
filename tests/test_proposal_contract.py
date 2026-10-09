import copy

import numpy as np
import pytest

from styne.backend import get_backend
from styne.mcmc import BlockProposal, MetropolisHastings, MetropolisedRandomWalk
from styne.mcmc.diagnostics import DummyDiagnostics, PersistentAcceptanceRateDiagnostics
from styne.mcmc.method.mlda import MLDAProposal
from styne.mcmc.method.mrw import MRWProposal
from styne.mcmc.method.pcn import PCNProposal
from styne.mcmc.proposal import ProposalMethod
from styne.mcmc.surrogate import SurrogateTransitionMeasure
from styne.parameter import Vector
from styne.statistics import Gaussian, IIDCovarianceMatrix
from styne.statistics.radonnikodym import RadonNikodym
from styne.utility.partition import Partition, PartitionRule


def gaussian(variance, mean):
    return Gaussian(IIDCovarianceMatrix(len(mean), variance), Vector(np.array(mean)))


def test_conditioned_proposal_supplies_the_reference_used_for_acceptance():
    target = gaussian(1.0, [0.0]).density
    reference = gaussian(2.0, [1.0])
    calls = []

    class ConditionedProposal(ProposalMethod):
        def condition(self, state):
            calls.append(state)
            return PCNProposal(reference, 0.5)

        def propose(self, state, rng):
            raise AssertionError('Only the conditioned proposal may draw.')

    sampler = MetropolisHastings(target, ConditionedProposal(), DummyDiagnostics())
    state = sampler.initial_state(Vector(np.array([0.4])))
    _, transition, _ = sampler.step(state, np.random.default_rng(5))
    proposed = transition.proposed.parameter
    ratio = target.evaluate_log(proposed) - target.evaluate_log(state.parameter)
    ratio -= reference.density.evaluate_log(proposed)
    ratio += reference.density.evaluate_log(state.parameter)

    assert calls == [state.parameter]
    assert transition.logAcceptanceProbability == pytest.approx(min(0.0, ratio))


def test_retargeting_rebuilds_implicit_reference_and_keeps_explicit_reference():
    first = RadonNikodym(gaussian(1.0, [0.0]), gaussian(1.0, [1.0]).density)
    second = RadonNikodym(gaussian(2.0, [2.0]), gaussian(0.5, [0.0]).density)
    explicit = gaussian(3.0, [-1.0])
    for reference in (first.reference, explicit):
        sampler = MetropolisHastings(first, PCNProposal(reference, 0.5),
                                     DummyDiagnostics())
        changed = sampler.with_target(second)
        assert sampler.proposal.reference is reference
        expected = second.reference if reference is first.reference else explicit
        assert changed.proposal.reference is expected
        assert sampler.target is first


@pytest.mark.parametrize('backendName', ['numpy', 'jax', 'pytorch'])
def test_block_proposals_propagate_rng_and_sum_reference_corrections(backendName):
    if backendName != 'numpy':
        pytest.importorskip('torch' if backendName == 'pytorch' else backendName)
    backend = get_backend(backendName)
    rule = PartitionRule([[0], [1]])
    start = Vector(backend.asarray([0.1, 0.2]))
    prior = Gaussian(IIDCovarianceMatrix(1, backend.asarray(1.0)),
                     Vector(backend.asarray([0.0])))
    proposals = [PCNProposal(prior, 0.5), MRWProposal(prior.covariance)]
    proposal = BlockProposal(Partition(rule, start), proposals)

    record, _ = proposal.propose(start, backend.random_state(4))
    rng = backend.random_state(4)
    coordinates = []
    for index, part in enumerate(proposals):
        transition, rng = part.propose(Vector(rule.extract(index, start.coordinate)), rng)
        coordinates.append(transition.proposed.parameter.coordinate)
    np.testing.assert_allclose(np.asarray(record.proposed.parameter.coordinate),
                               np.asarray(rule.merge(coordinates)))
    current = Vector(rule.extract(0, start.coordinate))
    proposed = Vector(rule.extract(0, record.proposed.parameter.coordinate))
    expected = prior.density.evaluate_log(current) - prior.density.evaluate_log(proposed)
    np.testing.assert_allclose(np.asarray(record.auxiliary['logCorrection']),
                               np.asarray(expected), atol=1e-6)


def test_sampler_copies_do_not_share_numpy_random_state():
    target = gaussian(1.0, [0.0]).density
    sampler = MetropolisedRandomWalk(target, IIDCovarianceMatrix(1, 1.0),
                                     DummyDiagnostics(), rng=np.random.default_rng(7))
    rngState = copy.deepcopy(sampler._rng.bit_generator.state)
    replacement = sampler.with_proposal(MRWProposal(IIDCovarianceMatrix(1, 0.5)))
    replacement.run(3, Vector(np.zeros(1)))
    assert sampler._rng.bit_generator.state == rngState


@pytest.mark.parametrize('indices', [
    [], [[0.5], [1]], [[True], [False]], [[[0]], [[1]]],
    [[0], [0]], [[0], [2]], [[-1], [0]],
])
def test_invalid_partition_indices_are_rejected(indices):
    with pytest.raises(ValueError):
        PartitionRule(indices)


@pytest.mark.parametrize('backendName', ['numpy', 'jax', 'pytorch'])
def test_partition_roundtrip_preserves_batches_and_coordinate_order(backendName):
    if backendName != 'numpy':
        pytest.importorskip('torch' if backendName == 'pytorch' else backendName)
    backend = get_backend(backendName)
    coordinate = backend.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rule = PartitionRule([[2, 0], [1]])
    result = rule.merge([rule.extract(i, coordinate) for i in range(2)])
    assert type(result) is type(coordinate)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(coordinate))


def test_new_outer_runner_does_not_reset_shared_inner_diagnostics():
    target = gaussian(1.0, [0.0]).density
    inner = MetropolisedRandomWalk(target, IIDCovarianceMatrix(1, 0.5),
                                  PersistentAcceptanceRateDiagnostics())
    start = Vector(np.zeros(1))
    inner.run(3, start)
    summary = inner.diagnostics.summary()
    proposal = MLDAProposal(SurrogateTransitionMeasure(inner, 2))
    outer = MetropolisHastings(target, proposal, DummyDiagnostics())

    for copied in (outer.with_target(target), outer.with_proposal(proposal)):
        assert copied.chain.length == 0
        assert inner.diagnostics.summary() == summary
        assert inner.diagnostics._total == 3


def test_new_runner_resets_its_own_persistent_diagnostics():
    target = gaussian(1.0, [0.0]).density
    original = MetropolisedRandomWalk(target, IIDCovarianceMatrix(1, 0.5),
                                     PersistentAcceptanceRateDiagnostics())
    original.run(3, Vector(np.zeros(1)))
    copied = original.with_proposal(original.proposal)
    assert copied.diagnostics._total == 0
    assert original.diagnostics._total == 3
