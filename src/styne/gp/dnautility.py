import itertools

import numpy as np

from enum import Enum
from scipy.fft import dct, dst

from styne.utility.partition import PartitionRule, Partition
from styne.statistics.gaussian import Gaussian
from styne.parameter.vector import Vector
from styne.statistics.covariance import DiagonalCovarianceMatrix


def cos_series(a, axis=-1, out=None):
    """
    Evaluate the cosine series, vectorising along 'axis'. The optional 'out' buffer 
    eliminates array allocations in MCMC hot-loops; use with 'overwrite_x=True' for 
    potential in-place transform speedups.
    """
    scale_factor = 1.0 / np.sqrt(2.)
    
    if out is None:
        acopy = np.array(a * scale_factor)
        sl0 = [slice(None)] * a.ndim
        sl0[axis] = 0
        acopy[tuple(sl0)] *= np.sqrt(2.)
        
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (0, 1)
        acopy_padded = np.pad(acopy, pad_width)
        return dct(acopy_padded, norm="backward", type=1, axis=axis)
    
    # Buffered path
    # 'out' is presumed to be the correctly sized (n+1) grid buffer
    sl_all = [slice(None)] * a.ndim
    sl_main = list(sl_all)
    sl_main[axis] = slice(0, -1)
    
    np.multiply(a, scale_factor, out=out[tuple(sl_main)])
    
    sl_first = list(sl_all)
    sl_first[axis] = 0
    out[tuple(sl_first)] *= np.sqrt(2.)
    
    sl_last = list(sl_all)
    sl_last[axis] = -1
    out[tuple(sl_last)] = 0.0
    
    # We use overwrite_x=True to encourage in-place FFT if supported
    return dct(out, norm="backward", type=1, axis=axis, overwrite_x=True)


def sin_series(a, axis=-1, out=None):
    """
    Evaluate the sine series, vectorising along 'axis'. The optional 'out' buffer 
    minimises GC pressure by reusing grid-sized memory across MCMC steps.
    """
    scale_factor = 1.0 / np.sqrt(2.0)
    
    if out is None:
        sl1 = [slice(None)] * a.ndim
        sl1[axis] = slice(1, None)
        
        # dst type 1 on interior points
        seriesInner = dst(a[tuple(sl1)] * scale_factor, norm="backward", type=1, axis=axis)
        
        # pad both sides
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (1, 1)
        return np.pad(seriesInner, pad_width)

    # Buffered path
    # 'out' is presumed to be the correctly sized (n+1) grid buffer
    sl_all = [slice(None)] * a.ndim
    
    sl_first = list(sl_all)
    sl_first[axis] = 0
    out[tuple(sl_first)] = 0.0
    
    sl_last = list(sl_all)
    sl_last[axis] = -1
    out[tuple(sl_last)] = 0.0
    
    sl_interior = list(sl_all)
    sl_interior[axis] = slice(1, -1)
    
    sl_a_in = list(sl_all)
    sl_a_in[axis] = slice(1, None)
    
    np.multiply(a[tuple(sl_a_in)], scale_factor, out=out[tuple(sl_interior)])
    
    # Update interior in-place
    res = dst(out[tuple(sl_interior)], norm="backward", type=1, axis=axis, overwrite_x=True)
    out[tuple(sl_interior)] = res
    
    return out


def sin_series_rows(a):
    """Sine series along rows (axis 1)."""
    sl = [slice(None)] * a.ndim
    sl[1] = slice(1, None)
    inner = dst(
        a[tuple(sl)] / np.sqrt(2.),
        norm="backward", type=1, axis=1
    )
    shape = list(inner.shape)
    shape[1] += 2
    out = np.zeros(shape)
    outSlice = [slice(None)] * a.ndim
    outSlice[1] = slice(1, -1)
    out[tuple(outSlice)] = inner
    return out


def sin_series_cols(a):
    """Sine series along columns (axis 0)."""
    sl = [slice(None)] * a.ndim
    sl[0] = slice(1, None)
    inner = dst(
        a[tuple(sl)] / np.sqrt(2.),
        norm="backward", type=1, axis=0
    )
    shape = list(inner.shape)
    shape[0] += 2
    out = np.zeros(shape)
    outSlice = [slice(None)] * a.ndim
    outSlice[0] = slice(1, -1)
    out[tuple(outSlice)] = inner
    return out


def cos_series_rows(a):
    """Cosine series along rows (axis 1)."""
    b = a / np.sqrt(2.)
    sl = [slice(None)] * a.ndim
    sl[1] = 0
    b[tuple(sl)] *= np.sqrt(2.)
    shape = list(b.shape)
    shape[1] += 1
    padded = np.zeros(shape)
    paddedSlice = [slice(None)] * a.ndim
    paddedSlice[1] = slice(0, b.shape[1])
    padded[tuple(paddedSlice)] = b
    return dct(padded, norm="backward", type=1, axis=1)


def cos_series_cols(a):
    """Cosine series along columns (axis 0)."""
    b = a / np.sqrt(2)
    sl = [slice(None)] * a.ndim
    sl[0] = 0
    b[tuple(sl)] *= np.sqrt(2.)
    shape = list(b.shape)
    shape[0] += 1
    padded = np.zeros(shape)
    paddedSlice = [slice(None)] * a.ndim
    paddedSlice[0] = slice(0, b.shape[0])
    padded[tuple(paddedSlice)] = b
    return dct(padded, norm="backward", type=1, axis=0)


def adj_cos_series_1d(r, q, axis=0):
    """Adjoint of cos_series: R^{q+2} -> R^{q+1}."""
    fullDct = dct(r, norm="backward", type=1, axis=axis)
    
    sl0 = [slice(None)] * r.ndim
    sl0[axis] = 0
    slq1 = [slice(None)] * r.ndim
    slq1[axis] = q + 1
    
    r0, rq1 = r[tuple(sl0)], r[tuple(slq1)]
    
    signs = (-1) ** np.arange(q + 1)
    # Broadcast signs to match r's shape excluding the transformed axis
    if r.ndim > 1:
        # Move axis to front temporarily for broadcasting or use reshape
        # simpler: signs.reshape(...)
        new_shape = [1] * r.ndim
        new_shape[axis] = q + 1
        boundary = r0 + signs.reshape(new_shape) * rq1
        
        scale = np.full(q + 1, np.sqrt(2.) / 2.)
        scale[0] = 0.5
        
        res_sl = [slice(None)] * r.ndim
        res_sl[axis] = slice(0, q + 1)
        
        return scale.reshape(new_shape) * (fullDct[tuple(res_sl)] + boundary)
    else:
        boundary = r0 + signs * rq1
        scale = np.full(q + 1, np.sqrt(2.) / 2.)
        scale[0] = 0.5
        return scale * (fullDct[:q + 1] + boundary)


def adj_sin_series_1d(r, q, axis=0):
    """Adjoint of sin_series (with prepended 0): R^{q+2} -> R^q."""
    sl = [slice(None)] * r.ndim
    sl[axis] = slice(1, q + 1)
    return (np.sqrt(2.) / 2.) * dst(r[tuple(sl)], norm="backward", type=1, axis=axis)


def adj_cos_series_rows(m, q):
    """Adjoint of cos_series_rows (axis 1)."""
    fullDct = dct(m, norm="backward", type=1, axis=1)
    sl0 = [slice(None)] * m.ndim
    sl0[1] = 0
    slq1 = [slice(None)] * m.ndim
    slq1[1] = q + 1
    
    r0, rq1 = m[tuple(sl0)], m[tuple(slq1)]
    signs = (-1) ** np.arange(q + 1)
    
    new_shape = [1] * m.ndim
    new_shape[1] = q + 1
    signs_aligned = signs.reshape(new_shape)
    
    res_sl = [slice(None)] * m.ndim
    res_sl[1] = slice(0, q + 1)
    
    r0_exp = np.expand_dims(r0, 1)
    rq1_exp = np.expand_dims(rq1, 1)
    boundary = r0_exp + signs_aligned * rq1_exp
    
    scale = np.full(q + 1, np.sqrt(2.) / 2.)
    scale[0] = 0.5
    scale_aligned = scale.reshape(new_shape)
    
    return scale_aligned * (fullDct[tuple(res_sl)] + boundary)


def adj_cos_series_cols(m, q):
    """Adjoint of cos_series_cols (axis 0)."""
    fullDct = dct(m, norm="backward", type=1, axis=0)
    sl0 = [slice(None)] * m.ndim
    sl0[0] = 0
    slq1 = [slice(None)] * m.ndim
    slq1[0] = q + 1
    
    r0, rq1 = m[tuple(sl0)], m[tuple(slq1)]
    signs = (-1) ** np.arange(q + 1)
    new_shape = [1] * m.ndim
    new_shape[0] = q + 1
    
    r0_exp = np.expand_dims(r0, 0)
    rq1_exp = np.expand_dims(rq1, 0)
    boundary = r0_exp + signs.reshape(new_shape) * rq1_exp
    
    scale = np.full(q + 1, np.sqrt(2.) / 2.)
    scale[0] = 0.5
    scale_aligned = scale.reshape(new_shape)
    
    res_sl = [slice(None)] * m.ndim
    res_sl[0] = slice(0, q + 1)
    return scale_aligned * (fullDct[tuple(res_sl)] + boundary)


def adj_sin_series_rows(m, q):
    """Adjoint of sin_series_rows (axis 1)."""
    shape = list(m.shape)
    shape[1] = q + 1
    out = np.zeros(shape)
    
    sl_m = [slice(None)] * m.ndim
    sl_m[1] = slice(1, q + 1)
    
    sl_out = [slice(None)] * m.ndim
    sl_out[1] = slice(1, None)
    
    out[tuple(sl_out)] = (np.sqrt(2.) / 2.) * dst(
        m[tuple(sl_m)], norm="backward", type=1, axis=1)
    return out


def adj_sin_series_cols(m, q):
    """Adjoint of sin_series_cols (axis 0)."""
    shape = list(m.shape)
    shape[0] = q + 1
    out = np.zeros(shape)
    
    sl_m = [slice(None)] * m.ndim
    sl_m[0] = slice(1, q + 1)
    
    sl_out = [slice(None)] * m.ndim
    sl_out[0] = slice(1, None)
    
    out[tuple(sl_out)] = (np.sqrt(2.) / 2.) * dst(
        m[tuple(sl_m)], norm="backward", type=1, axis=0)
    return out


class BC(Enum):
    """
    Boundary condition type for one axis of a DNA Fourier component.

    NEUMANN corresponds to a cosine expansion (b_i = 0 in the paper's
    notation). DIRICHLET corresponds to a sine expansion (b_i = 1). Content
    here was previously inline comments on the enum values, moved in per
    your instruction to catch these.
    """
    NEUMANN = 0
    DIRICHLET = 1


class BoundaryCondition:
    r"""
    Boundary condition vector $b \in \{BC\}^d$ for a single DNA component.

    Parameters
    ----------
    bcs : tuple
        One `BC` value per spatial dimension.
    """

    def __init__(self, bcs: tuple):
        self._bcs = tuple(bcs)

    @property
    def d(self) -> int:
        return len(self._bcs)

    def __getitem__(self, i: int) -> BC:
        return self._bcs[i]

    def block_size(self, q) -> int:
        q = (q,) * self.d if np.isscalar(q) else tuple(q)
        size = 1
        for bc_i, q_i in zip(self._bcs, q):
            size *= (q_i + 1) if bc_i == BC.NEUMANN else q_i
        return size

    @classmethod
    def all_combinations(cls, d: int) -> list:
        return [cls(bcs)
                for bcs in itertools.product([BC.NEUMANN, BC.DIRICHLET], repeat=d)]


class DNACoarseFineSplit(PartitionRule):
    """Bookkeeping for the coarse/fine partition of DNA spectral coefficients.

    Parameters
    ----------
    q : int | tuple
        Full DNA resolution (interior DOF per axis).
    qC : int | tuple
        Coarse resolution threshold; modes with stored index < threshold are coarse.
    d : int
        Spatial dimension (1 or 2).
    """

    def __init__(self, q, qC, d: int):

        q = (q,) * d if np.isscalar(q) else tuple(q)
        qC = (qC,) * d if np.isscalar(qC) else tuple(qC)

        if any(qc_j >= q_j for qc_j, q_j in zip(qC, q)):
            raise ValueError(f"qC {qC} must be less than q {q} on every axis.")

        self._q = q
        self._qC = qC
        self._d = d

        coarseIdx = []
        fineIdx = []
        offset = 0

        for bc in BoundaryCondition.all_combinations(d):
            blockSize = bc.block_size(q)

            if d == 1:
                nCoarse = qC[0] + 1 if bc[0] == BC.NEUMANN else qC[0]
                coarseIdx.extend(range(offset, offset + nCoarse))
                fineIdx.extend(range(offset + nCoarse, offset + blockSize))
            else:
                nCoarseX = qC[0] + 1 if bc[0] == BC.NEUMANN else qC[0]
                nCoarseY = qC[1] + 1 if bc[1] == BC.NEUMANN else qC[1]
                nModeX = q[0] + 1 if bc[0] == BC.NEUMANN else q[0]
                nModeY = q[1] + 1 if bc[1] == BC.NEUMANN else q[1]

                for i in range(nModeX):
                    for j in range(nModeY):
                        idx = offset + i * nModeY + j
                        if i < nCoarseX and j < nCoarseY:
                            coarseIdx.append(idx)
                        else:
                            fineIdx.append(idx)

            offset += blockSize

        self._indices = [
            np.array(coarseIdx, dtype=int),
            np.array(fineIdx, dtype=int)
        ]

    @property
    def numComponents(self) -> int:
        return 2

    def indices(self, idx: int) -> np.ndarray:
        return self._indices[idx]


class DNACoarseFinePartition(Partition):
    """
    Partition object corresponding to the DNA coarse-fine mode split.

    Parameters
    ----------
    dnaGP : GaussianProcess
        Source GP the partition is built from. Duck-typed, not enforced by
        `isinstance`, checked at construction only via `hasattr(dnaGP,
        'resolution')`.
    qC : int | tuple
        Coarse resolution threshold, same inference caveat as `q` elsewhere.
    d : int
        Spatial dimension (1 or 2).
    """

    def __init__(self, dnaGP, qC, d: int):

        if not hasattr(dnaGP, 'resolution'):
            raise AttributeError("GaussianProcess must expose 'resolution' for DNA partition.")

        super().__init__(DNACoarseFineSplit(dnaGP.resolution, qC, d), dnaGP.parameter)
        self._measure = dnaGP.measure

    def coarse_measure(self):
        return self._component_measure(0)

    def fine_measure(self):
        return self._component_measure(1)

    def _component_measure(self, idx: int):

        cov = self._measure.covariance
        if not isinstance(cov, DiagonalCovarianceMatrix):
            raise TypeError("Extraction only supported for DiagonalCovarianceMatrix")

        compVar = self._rule.extract(idx, cov.marginalVariance)
        compMean = self._rule.extract(idx, self._measure.mean.coordinate)

        return Gaussian(DiagonalCovarianceMatrix(compVar), Vector(compMean))
