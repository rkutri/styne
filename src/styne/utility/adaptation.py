"""Proposal-scale adaptation during a separate warm-up trajectory."""
from typing import Any, NamedTuple

import numpy as np

from styne.backend import infer_backend
from styne.mcmc.metropolishastings import MetropolisHastings
from styne.mcmc.method.mrw import MRWProposal
from styne.mcmc.method.pcn import PCNProposal
from styne.statistics.covariance import IIDCovarianceMatrix


class AdaptationState(NamedTuple):
    """Numerical warm-up state, including the proposal scale and step count."""

    chainState: Any
    logScale: Any
    stepCount: Any

    @property
    def parameter(self):
        return self.chainState.parameter


class RobbinsMonroAdaptation:
    """Adapt an IID random-walk variance or pCN step size during warm-up.

    The log scale moves by (i + offset)^(-decay) (a - targetAcceptance),
    where i counts transitions and a is the accepted fraction of the batch.
    A batch shares one scale; each chain has its own acceptance decision.
    pCN step sizes are capped at 1. The sampler's acceptance strategy is kept.

    'freeze' returns a sampler with the final scale held constant. Warm-up
    states are not appended to its production chain. This wrapper deliberately
    accepts only a single MRW or pCN proposal; it does not infer which nested
    proposal or acceptance rate a caller intends to adapt.
    """

    def __init__(self, sampler: MetropolisHastings, targetAcceptance=0.3,
                 offset=100, decay=0.6):
        if not isinstance(sampler, MetropolisHastings):
            raise TypeError("Adaptation requires a MetropolisHastings sampler.")
        if not np.isfinite(targetAcceptance) or not 0 < targetAcceptance < 1:
            raise ValueError("targetAcceptance must be finite and between 0 and 1.")
        if not np.isfinite(offset) or offset < 0:
            raise ValueError("offset must be finite and non-negative.")
        if not np.isfinite(decay) or not 0.5 < decay <= 1:
            raise ValueError("decay must be finite and in (0.5, 1].")

        proposal = sampler.proposal
        if isinstance(proposal, PCNProposal):
            initialScale = proposal.beta
        elif isinstance(proposal, MRWProposal) and isinstance(
                proposal.covariance, IIDCovarianceMatrix):
            covariance = proposal.covariance
            initialScale = covariance.scaling * covariance.marginalVariance[0]
        else:
            raise TypeError("Adaptation requires an IID MRW or a pCN proposal.")

        self._sampler = sampler
        self._initialScale = initialScale
        self._targetAcceptance = targetAcceptance
        self._offset = offset
        self._decay = decay

    @property
    def sampler(self):
        return self._sampler

    def initial_state(self, parameter) -> AdaptationState:
        backend = infer_backend(parameter.coordinate)
        metadata = backend.metadata(parameter.coordinate)
        scale = backend.asarray(
            self._initialScale, dtype=metadata.dtype, device=metadata.device
        )
        return AdaptationState(
            self._sampler.initial_state(parameter), backend.namespace.log(scale),
            backend.asarray(0, dtype=metadata.dtype, device=metadata.device),
        )

    def step(self, state: AdaptationState, rng):
        """Take one transition at the current scale, then update that scale."""
        sampler = self._sampler_at(state.logScale, rng)
        chainState, transition, nextRng = sampler.step(state.chainState, rng)
        backend = infer_backend(state.logScale)
        metadata = backend.metadata(state.logScale)
        one, zero = (
            backend.asarray(value, dtype=metadata.dtype, device=metadata.device)
            for value in (1.0, 0.0)
        )
        acceptance = backend.namespace.mean(
            backend.namespace.where(transition.outcome, one, zero)
        )
        stepCount = state.stepCount + 1
        gain = (stepCount + self._offset) ** (-self._decay)
        logScale = state.logScale + gain * (acceptance - self._targetAcceptance)
        if isinstance(self._sampler.proposal, PCNProposal):
            logScale = backend.namespace.minimum(logScale, zero)
        return AdaptationState(chainState, logScale, stepCount), transition, nextRng

    def run(self, nSteps: int, state: AdaptationState, rng):
        """Advance warm-up, returning its numerical state and random state."""
        if isinstance(nSteps, bool) or not isinstance(nSteps, int) or nSteps < 0:
            raise ValueError("nSteps must be a non-negative integer.")
        if nSteps == 0:
            return state, rng
        backend = infer_backend(state.parameter.coordinate)
        if not backend.capabilities.transformedLoops:
            for _ in range(nSteps):
                state, _, rng = self.step(state, rng)
            return state, rng

        def advance(carry, _):
            current, currentRng = carry
            nextState, _, nextRng = self.step(current, currentRng)
            return (nextState, nextRng), None

        def execute(current, currentRng):
            return backend.scan(
                advance, (current, currentRng), None, length=nSteps
            )

        (state, rng), _ = backend.compile(execute)(state, rng)
        return state, rng

    def freeze(self, state: AdaptationState, rng=None) -> MetropolisHastings:
        """Fix the scale; pass the returned warm-up RNG to continue with 'run'."""
        return self._sampler_at(state.logScale, rng)

    def _sampler_at(self, logScale, rng=None):
        scale = infer_backend(logScale).namespace.exp(logScale)
        proposal = self._sampler.proposal
        if isinstance(proposal, PCNProposal):
            proposal = PCNProposal(proposal.reference, scale)
        else:
            proposal = MRWProposal(IIDCovarianceMatrix(
                proposal.covariance.dimension, scale
            ))
        return self._sampler.with_proposal(proposal, rng=rng)
