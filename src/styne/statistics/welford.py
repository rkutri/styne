"""Online moments with a pure numerical core and an eager accumulator interface."""
from typing import Any, NamedTuple

import numpy as np

from styne.backend import BackendInferenceError, get_backend, infer_backend


class WelfordRecord(NamedTuple):
    """Count, mean and centred second-moment sum, diagonal or full."""

    nData: Any
    mean: Any
    m2: Any


class WelfordAccumulator:
    """Online mean and marginal variance using Welford's stable recurrence.

    'initial_record' and 'update_record' are pure and backend-native. The eager
    'update', 'mean', 'marginal_variance' and 'reset' interface is retained.
    """

    def __init__(self):
        self._record = None

    @staticmethod
    def initial_record(realisation):
        """Initialise moments without counting a sample; use floating observations."""
        try:
            backend = infer_backend(realisation)
        except BackendInferenceError:
            backend = get_backend('numpy')
            realisation = backend.asarray(realisation)
        metadata = backend.metadata(realisation)
        mean = backend.zeros(realisation.shape, dtype=metadata.dtype,
                             device=metadata.device)
        m2 = backend.zeros(realisation.shape, dtype=metadata.dtype,
                           device=metadata.device)
        return WelfordRecord(
            backend.asarray(0, dtype='int32', device=metadata.device), mean, m2,
        )

    @staticmethod
    def update_record(record, realisation):
        """Return moments including one observation, without changing the input."""
        backend = infer_backend(record.mean)
        metadata = backend.metadata(record.mean)
        nData = record.nData + 1
        count = backend.asarray(nData, dtype=metadata.dtype, device=metadata.device)
        delta = realisation - record.mean
        mean = record.mean + delta / count
        m2 = record.m2 + delta * (realisation - mean)
        return WelfordRecord(nData, mean, m2)

    def mean(self):
        return None if self._record is None else self._record.mean

    def marginal_variance(self):
        """Return the unbiased marginal variance after at least two samples."""
        if self.nData < 2:
            raise RuntimeError("Insufficient data for variance estimation.")
        backend = infer_backend(self._record.mean)
        metadata = backend.metadata(self._record.mean)
        denominator = backend.asarray(self._record.nData - 1,
                                      dtype=metadata.dtype, device=metadata.device)
        return self._record.m2 / denominator

    def condition_number(self):
        margVar = self.marginal_variance()
        if np.min(margVar) < 1e-12:
            raise RuntimeError("Singular marginal variance.")
        return np.max(margVar) / np.min(margVar)

    @property
    def nData(self):
        return 0 if self._record is None else int(self._record.nData)

    def update(self, realisation):
        if self._record is None:
            self._record = self.initial_record(realisation)
        self._record = self.update_record(self._record, realisation)

    def reset(self):
        self._record = None


class FullCovarianceAccumulator:
    """Online mean and full sample covariance of real observation vectors.

    'initial_record' and 'update_record' carry immutable, backend-native state
    for adaptation and compiled loops. Each observation has shape (dimension,)
    and 'm2' has shape (dimension, dimension). No regularisation is added.
    """

    def __init__(self):
        self._record = None

    @staticmethod
    def initial_record(realisation):
        """Create empty moments from a non-empty vector, without counting it."""
        try:
            backend = infer_backend(realisation)
        except BackendInferenceError:
            backend = get_backend('numpy')
        realisation = backend.asarray(realisation)
        if realisation.ndim != 1 or realisation.shape[0] == 0:
            raise ValueError("Expected a non-empty one-dimensional observation.")
        # Promote integer observations before creating a fixed-dtype loop state.
        metadata = backend.metadata(realisation / 1.0)
        dimension = realisation.shape[0]
        mean = backend.zeros((dimension,), dtype=metadata.dtype,
                             device=metadata.device)
        m2 = backend.zeros((dimension, dimension), dtype=metadata.dtype,
                           device=metadata.device)
        return WelfordRecord(
            backend.asarray(0, dtype='int32', device=metadata.device), mean, m2,
        )

    @staticmethod
    def update_record(record, realisation):
        """Include one observation without changing the input record or vector."""
        backend = infer_backend(record.mean)
        metadata = backend.metadata(record.mean)
        realisation = backend.asarray(realisation, dtype=metadata.dtype,
                                      device=metadata.device)

        nData = record.nData + 1
        count = backend.asarray(nData, dtype=metadata.dtype, device=metadata.device)
        delta = realisation - record.mean
        mean = record.mean + delta / count
        # Scale first to avoid squaring a large offset on the first observation.
        scaledDelta = backend.namespace.sqrt((count - 1) / count) * delta
        m2 = record.m2 + scaledDelta[:, None] * scaledDelta[None, :]
        return WelfordRecord(nData, mean, m2)

    def mean(self):
        return None if self._record is None else self._record.mean

    def covariance(self):
        """Return the sample covariance (ddof=1) after at least two observations."""
        if self.nData < 2:
            raise RuntimeError("Insufficient data for covariance estimation.")
        backend = infer_backend(self._record.mean)
        metadata = backend.metadata(self._record.mean)
        denominator = backend.asarray(self._record.nData - 1,
                                      dtype=metadata.dtype, device=metadata.device)
        return self._record.m2 / denominator

    def marginal_variance(self):
        """Return the diagonal of the sample covariance."""
        covariance = self.covariance()
        return infer_backend(covariance).namespace.diagonal(covariance)

    @property
    def nData(self):
        return 0 if self._record is None else int(self._record.nData)

    def update(self, realisation):
        if self._record is None:
            self._record = self.initial_record(realisation)
        self._record = self.update_record(self._record, realisation)

    def reset(self):
        self._record = None
