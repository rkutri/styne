import numpy as np
import pytest

from scipy.sparse.linalg import spsolve

from styne.utility.finiteElement import (
    p1_stiffness_1d, p1_mass_lumped_1d, apply_dirichlet_1d,
    q1_stiffness_2d, q1_mass_lumped_2d, apply_dirichlet_2d,
)


def test_p1_1d_dirichlet():
    """Solve −u'' = π² sin(πx) on [0,1] with Dirichlet BCs; exact: sin(πx).

    P1 elements on uniform mesh h = 1/64.  Error bound: C·h² with C = 1.
    """
    n = 64
    vertices = np.linspace(0., 1., n + 1)
    h = 1. / n

    K = p1_stiffness_1d(vertices, np.ones(n))
    M = p1_mass_lumped_1d(vertices)

    # Load vector via nodal quadrature: b_i ≈ M_ii · f(x_i)
    f = np.pi ** 2 * np.sin(np.pi * vertices)
    b = M.diagonal() * f

    K_bc, b_bc = apply_dirichlet_1d(K, b)
    u = spsolve(K_bc, b_bc)

    u_exact = np.sin(np.pi * vertices)
    err = np.max(np.abs(u - u_exact))

    assert err < h ** 2, (
        f"P1 1D: max nodal error {err:.2e} exceeds bound {h**2:.2e}"
    )


def test_q1_2d_dirichlet():
    """Solve −Δu = 2π² sin(πx)sin(πy) on [0,1]²; exact: sin(πx)sin(πy).

    Q1 elements on uniform mesh h = 1/32.  Error bound: C·h² with C = 0.5.
    """
    n = 32
    xVert = np.linspace(0., 1., n + 1)
    yVert = np.linspace(0., 1., n + 1)
    h = 1. / n

    K = q1_stiffness_2d(xVert, yVert, np.ones(n * n))
    M = q1_mass_lumped_2d(xVert, yVert)

    XX, YY = np.meshgrid(xVert, yVert, indexing='ij')
    f = 2. * np.pi ** 2 * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    b = M.diagonal() * f.ravel()

    K_bc, b_bc = apply_dirichlet_2d(K, b, n, n)
    u = spsolve(K_bc, b_bc)

    u_exact = (np.sin(np.pi * XX) * np.sin(np.pi * YY)).ravel()
    err = np.max(np.abs(u - u_exact))

    assert err < 4. * h ** 2, (
        f"Q1 2D: max nodal error {err:.2e} exceeds bound {4.*h**2:.2e}"
    )
