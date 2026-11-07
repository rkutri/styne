"""Online moments with a pure numerical core and an eager accumulator interface."""
from typing import Any, NamedTuple

import numpy as np

from styne.backend import BackendInferenceError, get_backend, infer_backend


class WelfordRecord(NamedTuple):
    """Count, mean and centred sum of squares for each coordinate."""

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
