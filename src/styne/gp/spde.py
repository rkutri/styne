import numpy as np

from numpy.random import Generator
from scipy.sparse.linalg import splu

from styne.statistics.measure import ProbabilityMeasure
from styne.utility.finiteElement import (
    p1_stiffness_1d, p1_mass_lumped_1d,
    q1_stiffness_2d, q1_mass_lumped_2d
)


class SPDEEngine1D(ProbabilityMeasure):
    """
    Sample from a Matérn GP in 1D via the SPDE (β=1).

    Solves (κ²M + K)u = τ M^½ ξ with P1 FEM, where M is the lumped mass
    matrix, K is the stiffness matrix with unit diffusion, and ξ ~ N(0, I)
    on the free DOFs.

    Parameters
    ----------
    vertices : array_like, shape (n,)
    kappa : float
        Inverse correlation length parameter.
    tau : float
        Scaling parameter (controls marginal variance together with κ).
    bc : int
        Boundary condition type following the DNA convention:
        0 = homogeneous Neumann, 1 = homogeneous Dirichlet (default).
    """

    def __init__(self, vertices, kappa: float, tau: float = 1., bc: int = 1):
        vertices = np.asarray(vertices, dtype=float)
        n = len(vertices)

        K = p1_stiffness_1d(vertices, np.ones(n - 1))
        M = p1_mass_lumped_1d(vertices)
        S = kappa**2 * M + K

        m_diag = M.diagonal()

        if bc == 1:
            self._free = np.arange(1, n - 1)
        else:
            self._free = np.arange(n)

        S_free = S[self._free, :][:, self._free]
        self._M_sqrt = np.sqrt(m_diag[self._free])
        self._tau = tau
        self._n = n
        self._lu = splu(S_free.tocsc())

    def draw(self, rng: Generator) -> np.ndarray:
        xi = rng.standard_normal(len(self._free))
        b = self._tau * self._M_sqrt * xi
        u_free = self._lu.solve(b)
        u = np.zeros(self._n)
        u[self._free] = u_free
        return u


class SPDEEngine2D(ProbabilityMeasure):
    """
    Sample from a Matérn GP in 2D via the SPDE (β=1).

    Solves (κ²M + K)u = τ M^½ ξ with Q1 FEM on a 2D grid.

    Parameters
    ----------
    xVert : array_like, shape (nx+1,)
    yVert : array_like, shape (ny+1,)
    kappa : float
    tau : float
    bc : tuple[int, int]
        Per-dimension BC flags (0 = Neumann, 1 = Dirichlet).
        Default (1, 1) applies homogeneous Dirichlet on all four sides.
    """

    def __init__(self, xVert, yVert, kappa: float, tau: float = 1.,
                 bc: tuple = (1, 1)):
        xVert = np.asarray(xVert, dtype=float)
        yVert = np.asarray(yVert, dtype=float)
        nx, ny = xVert.size - 1, yVert.size - 1
        nNodes = (nx + 1) * (ny + 1)

        K = q1_stiffness_2d(xVert, yVert, np.ones(nx * ny))
        M = q1_mass_lumped_2d(xVert, yVert)
        S = kappa**2 * M + K

        m_diag = M.diagonal()

        fixed = set()
        if bc[0] == 1:
            for j in range(ny + 1):
                fixed.add(j)
                fixed.add(nx * (ny + 1) + j)
        if bc[1] == 1:
            for i in range(nx + 1):
                fixed.add(i * (ny + 1))
                fixed.add(i * (ny + 1) + ny)

        free_mask = np.ones(nNodes, dtype=bool)
        for idx in fixed:
            free_mask[idx] = False
        self._free = np.where(free_mask)[0]

        S_free = S[self._free, :][:, self._free]
        self._M_sqrt = np.sqrt(m_diag[self._free])
        self._tau = tau
        self._nNodes = nNodes
        self._lu = splu(S_free.tocsc())

    def draw(self, rng: Generator) -> np.ndarray:
        xi = rng.standard_normal(len(self._free))
        b = self._tau * self._M_sqrt * xi
        u_free = self._lu.solve(b)
        u = np.zeros(self._nNodes)
        u[self._free] = u_free
        return u


class DNASPDEEngine1D(ProbabilityMeasure):
    """
    SPDE-based GP sampler with Dirichlet-Neumann Averaging in 1D.

    Averages the 2^1 = 2 independent SPDE solutions corresponding to all
    combinations of homogeneous Neumann (bc=0) and Dirichlet (bc=1) boundary
    conditions. Per Corollary 3.8 of Kutri & Scheichl (2026), this eliminates
    boundary artefacts entirely, yielding a genuinely isotropic random field
    whose covariance is the periodised covariance φ^(π)_{2α}(x−y). The
    remaining periodisation error decays exponentially in α (Lemma 3.3);
    α = 1 is sufficient in practice.
    """

    def __init__(self, vertices, kappa: float, tau: float = 1.):
        self._engines = [
            SPDEEngine1D(vertices, kappa, tau, bc) for bc in (0, 1)
        ]

    def draw(self, rng: Generator) -> np.ndarray:
        return sum(e.draw(rng) for e in self._engines) / np.sqrt(2.)


class DNASPDEEngine2D(ProbabilityMeasure):
    """
    SPDE-based GP sampler with Dirichlet-Neumann Averaging in 2D.

    Averages the 2^2 = 4 independent SPDE solutions corresponding to all
    combinations of Neumann/Dirichlet BCs per dimension. Per Corollary 3.8
    of Kutri & Scheichl (2026), this eliminates boundary artefacts entirely,
    yielding a genuinely isotropic random field.
    """

    def __init__(self, xVert, yVert, kappa: float, tau: float = 1.):
        bcs = [(bx, by) for bx in (0, 1) for by in (0, 1)]
        self._engines = [
            SPDEEngine2D(xVert, yVert, kappa, tau, bc) for bc in bcs
        ]

    def draw(self, rng: Generator) -> np.ndarray:
        return sum(e.draw(rng) for e in self._engines) * 0.5
