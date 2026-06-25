import numpy as np
import pytest
from numpy.random import default_rng
from scipy.fft import dct, dst

from styne.gp.dnautility import (
    sin_series_rows, sin_series_cols,
    cos_series_rows, cos_series_cols,
)


def _reference_sin_series_rows(a):
    """Original np.pad-based implementation for regression."""
    sl = [slice(None)] * a.ndim
    sl[1] = slice(1, None)
    inner = dst(
        a[tuple(sl)] / np.sqrt(2.),
        norm="backward", type=1, axis=1
    )
    pad = [(0, 0)] * a.ndim
    pad[1] = (1, 1)
    return np.pad(inner, pad)


def _reference_sin_series_cols(a):
    """Original np.pad-based implementation for regression."""
    sl = [slice(None)] * a.ndim
    sl[0] = slice(1, None)
    inner = dst(
        a[tuple(sl)] / np.sqrt(2.),
        norm="backward", type=1, axis=0
    )
    pad = [(0, 0)] * a.ndim
    pad[0] = (1, 1)
    return np.pad(inner, pad)


def _reference_cos_series_rows(a):
    """Original np.pad-based implementation for regression."""
    b = a / np.sqrt(2.)
    sl = [slice(None)] * a.ndim
    sl[1] = 0
    b[tuple(sl)] *= np.sqrt(2.)
    pad = [(0, 0)] * a.ndim
    pad[1] = (0, 1)
    return dct(np.pad(b, pad), norm="backward", type=1, axis=1)


def _reference_cos_series_cols(a):
    """Original np.pad-based implementation for regression."""
    b = a / np.sqrt(2)
    sl = [slice(None)] * a.ndim
    sl[0] = 0
    b[tuple(sl)] *= np.sqrt(2.)
    pad = [(0, 0)] * a.ndim
    pad[0] = (0, 1)
    return dct(np.pad(b, pad), norm="backward", type=1, axis=0)


class TestSinSeriesRows:
    def test_2d_matches_reference(self):
        rng = default_rng(101)
        a = rng.standard_normal((7, 6))
        ref = _reference_sin_series_rows(a.copy())
        got = sin_series_rows(a.copy())
        np.testing.assert_allclose(got, ref, atol=1e-14)

    def test_3d_matches_reference(self):
        rng = default_rng(102)
        a = rng.standard_normal((7, 6, 4))
        ref = _reference_sin_series_rows(a.copy())
        got = sin_series_rows(a.copy())
        np.testing.assert_allclose(got, ref, atol=1e-14)

    def test_boundary_zeros(self):
        rng = default_rng(103)
        a = rng.standard_normal((5, 4))
        out = sin_series_rows(a)
        np.testing.assert_allclose(out[:, 0], 0., atol=1e-15)
        np.testing.assert_allclose(out[:, -1], 0., atol=1e-15)


class TestSinSeriesCols:
    def test_2d_matches_reference(self):
        rng = default_rng(201)
        a = rng.standard_normal((6, 7))
        ref = _reference_sin_series_cols(a.copy())
        got = sin_series_cols(a.copy())
        np.testing.assert_allclose(got, ref, atol=1e-14)

    def test_3d_matches_reference(self):
        rng = default_rng(202)
        a = rng.standard_normal((6, 7, 3))
        ref = _reference_sin_series_cols(a.copy())
        got = sin_series_cols(a.copy())
        np.testing.assert_allclose(got, ref, atol=1e-14)

    def test_boundary_zeros(self):
        rng = default_rng(203)
        a = rng.standard_normal((4, 5))
        out = sin_series_cols(a)
        np.testing.assert_allclose(out[0, :], 0., atol=1e-15)
        np.testing.assert_allclose(out[-1, :], 0., atol=1e-15)


class TestCosSeriesRows:
    def test_2d_matches_reference(self):
        rng = default_rng(301)
        a = rng.standard_normal((7, 6))
        ref = _reference_cos_series_rows(a.copy())
        got = cos_series_rows(a.copy())
        np.testing.assert_allclose(got, ref, atol=1e-14)

    def test_3d_matches_reference(self):
        rng = default_rng(302)
        a = rng.standard_normal((7, 6, 5))
        ref = _reference_cos_series_rows(a.copy())
        got = cos_series_rows(a.copy())
        np.testing.assert_allclose(got, ref, atol=1e-14)


class TestCosSeriesCols:
    def test_2d_matches_reference(self):
        rng = default_rng(401)
        a = rng.standard_normal((6, 7))
        ref = _reference_cos_series_cols(a.copy())
        got = cos_series_cols(a.copy())
        np.testing.assert_allclose(got, ref, atol=1e-14)

    def test_3d_matches_reference(self):
        rng = default_rng(402)
        a = rng.standard_normal((6, 7, 3))
        ref = _reference_cos_series_cols(a.copy())
        got = cos_series_cols(a.copy())
        np.testing.assert_allclose(got, ref, atol=1e-14)
