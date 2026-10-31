"""Proposal-scale and full covariance adaptation for Metropolis-Hastings."""
from typing import Any, NamedTuple

import numpy as np

from styne.backend import infer_backend
from styne.mcmc.adaptation import Adaptation
from styne.mcmc.method.mrw import MRWProposal
from styne.mcmc.method.pcn import PCNProposal
from styne.statistics.covariance import DenseCovarianceMatrix, IIDCovarianceMatrix
from styne.statistics.welford import FullCovarianceAccumulator


class RobbinsMonroState(NamedTuple):
    """Numerical scale history; pCN's upper log-scale bound is zero."""

    logScale: Any
    stepCount: Any
    maxLogScale: Any


class RobbinsMonroAdaptation(Adaptation):
    """Adapt IID MRW variance or pCN beta from the batch's mean acceptance.

    The log scale moves by (i + offset)^(-decay) (a - targetAcceptance).
    Each chain retains its own acceptance decision. Adaptation continues unless
    'stopAfter' is set; the final proposal scale is then retained automatically.
    """

    def __init__(self, targetAcceptance=0.3, offset=100, decay=0.6, stopAfter=None):
        if not np.isfinite(targetAcceptance) or not 0 < targetAcceptance < 1:
            raise ValueError("targetAcceptance must be finite and between 0 and 1.")
        if not np.isfinite(offset) or offset < 0:
            raise ValueError("offset must be finite and non-negative.")
        if not np.isfinite(decay) or not 0.5 < decay <= 1:
            raise ValueError("decay must be finite and in (0.5, 1].")
        if stopAfter is not None and (
                isinstance(stopAfter, bool) or not isinstance(stopAfter, int)
                or stopAfter < 0):
            raise ValueError("stopAfter must be a non-negative integer or None.")
        self._targetAcceptance = targetAcceptance
        self._offset = offset
        self._decay = decay
        self._stopAfter = stopAfter

    def initial_state(self, parameter, proposalMethod):
        if isinstance(proposalMethod, PCNProposal):
            initialScale, maxLogScale = proposalMethod.beta, 0.0
        elif isinstance(proposalMethod, MRWProposal) and isinstance(
                proposalMethod.covariance, IIDCovarianceMatrix):
            covariance = proposalMethod.covariance
            initialScale = covariance.scaling * covariance.marginalVariance[0]
            maxLogScale = np.inf
        else:
            raise TypeError("Adaptation requires an IID MRW or a pCN proposal.")
        backend = infer_backend(parameter.coordinate)
        metadata = backend.metadata(parameter.coordinate)
        scale, count, maximum = (
            backend.asarray(value, dtype=metadata.dtype, device=metadata.device)
            for value in (initialScale, 0, maxLogScale)
        )
        return RobbinsMonroState(backend.namespace.log(scale), count, maximum)

    def prepare_proposal(self, proposalMethod, state):
        scale = infer_backend(state.logScale).namespace.exp(state.logScale)
        if isinstance(proposalMethod, PCNProposal):
            return PCNProposal(proposalMethod.reference, scale)
        return MRWProposal(IIDCovarianceMatrix(
            proposalMethod.covariance.dimension, scale
        ))

    def update_state(self, state, nextState, transitionData):
        backend = infer_backend(state.logScale)
        metadata = backend.metadata(state.logScale)
        one, zero = (
            backend.asarray(value, dtype=metadata.dtype, device=metadata.device)
            for value in (1.0, 0.0)
        )
        namespace = backend.namespace
        acceptance = namespace.mean(namespace.where(transitionData.outcome, one, zero))
        stepCount = state.stepCount + 1
        gain = (stepCount + self._offset) ** (-self._decay)
        logScale = namespace.minimum(
            state.logScale + gain * (acceptance - self._targetAcceptance),
            state.maxLogScale,
        )
        if self._stopAfter is not None:
            active = state.stepCount < self._stopAfter
            logScale = namespace.where(active, logScale, state.logScale)
            stepCount = namespace.where(active, stepCount, state.stepCount)
        return RobbinsMonroState(logScale, stepCount, state.maxLogScale)


class FullCovarianceAdaptationState(NamedTuple):
    """Empirical moments and the retained proposal's lower Cholesky factor."""

    moments: Any
    cholFactor: Any


class FullCovarianceAdaptation(Adaptation):
    """Learn one MRW chain's covariance, factorising only at scheduled updates.

    Every retained position updates the moments, including rejection repeats.
    After at least two observations, every 'interval'-th transition replaces
    the proposal covariance by the sample covariance plus 'nugget * I'.
    Adaptation continues until the sampler's optional 'freeze_adaptation'.
    """

    def __init__(self, interval=10, nugget=0.01):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("interval must be a positive integer.")
        if not np.isfinite(nugget) or nugget <= 0:
            raise ValueError("nugget must be finite and positive.")
        self._interval = interval
        self._nugget = nugget

    def initial_state(self, parameter, proposalMethod):
        if not isinstance(proposalMethod, MRWProposal):
            raise TypeError("Full covariance adaptation requires an MRW proposal.")
        moments = FullCovarianceAccumulator.initial_record(parameter.coordinate)
        factor = proposalMethod.covariance.to_cholesky()
        backend = infer_backend(parameter.coordinate, factor)
        metadata = backend.metadata(parameter.coordinate)
        factor = backend.asarray(factor, dtype=metadata.dtype, device=metadata.device)
        # PyTorch conditional branches also require matching array layouts.
        factor = factor.reshape((-1,)).reshape(factor.shape)
        return FullCovarianceAdaptationState(moments, factor)

    def prepare_proposal(self, proposalMethod, state):
        return MRWProposal(DenseCovarianceMatrix.from_cholesky(state.cholFactor))

    def update_state(self, state, nextState, transitionData):
        moments = FullCovarianceAccumulator.update_record(
            state.moments, nextState.parameter.coordinate,
        )
        backend = infer_backend(state.cholFactor)
        metadata = backend.metadata(state.cholFactor)
        due = (moments.nData >= 2) & (moments.nData % self._interval == 0)

        def refactor(operand):
            nData, m2, previousFactor = operand
            denominator = backend.asarray(
                nData - 1,
                dtype=metadata.dtype, device=metadata.device,
            )
            covariance = m2 / denominator
            identity = backend.eye(previousFactor.shape[-1], dtype=metadata.dtype,
                                   device=metadata.device)
            factor = backend.cholesky(covariance + self._nugget * identity)
            return factor.reshape((-1,)).reshape(factor.shape)

        # A conditional keeps Cholesky inside the scheduled branch under JAX.
        # Tensor-only operands and fresh outputs also support PyTorch's cond.
        factor = backend.cond(
            due, refactor, lambda operand: operand[2] * 1.0,
            (moments.nData, moments.m2, state.cholFactor),
        )
        return FullCovarianceAdaptationState(moments, factor)
