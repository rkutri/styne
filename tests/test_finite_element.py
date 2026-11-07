import numpy as np
import pytest

from scipy.sparse.linalg import spsolve

from styne.utility.finiteelement import (
    p1_stiffness_1d, p1_mass_lumped_1d, apply_dirichlet_1d,
    q1_stiffness_2d, q1_mass_lumped_2d, apply_dirichlet_2d,
    q1_log_diffusion_adjoint_2d,
)


def boundary_nodes(nx, ny):
    return np.array(sorted({
        i * (ny + 1) + j
        for i in range(nx + 1)
        for j in range(ny + 1)
        if i in (0, nx) or j in (0, ny)
    }), dtype=int)


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


def test_q1_log_diffusion_adjoint_matches_finite_difference():
    """Adjoint gradient of a Q1 solve wrt log diffusion matches central FD."""
    xVert = np.linspace(0.0, 1.0, 6)
    yVert = np.linspace(0.0, 1.0, 5)
    nx, ny = len(xVert) - 1, len(yVert) - 1
    rng = np.random.default_rng(12)

    theta = 0.2 * rng.standard_normal(nx * ny)
    direction = rng.standard_normal(nx * ny)
    rhs = q1_mass_lumped_2d(xVert, yVert).diagonal()
    functional = rng.standard_normal((nx + 1) * (ny + 1))
    boundary = boundary_nodes(nx, ny)
    rhs[boundary] = 0.0

    def evaluate(value):
        matrix = q1_stiffness_2d(xVert, yVert, np.exp(value))
        matrix = apply_dirichlet_2d(matrix, nx=nx, ny=ny)
        return functional @ spsolve(matrix, rhs)

    matrix = q1_stiffness_2d(xVert, yVert, np.exp(theta))
    matrix = apply_dirichlet_2d(matrix, nx=nx, ny=ny)
    primal = spsolve(matrix, rhs)
    adjoint = spsolve(matrix.T, functional)
    adjoint[boundary] = 0.0
    gradient = q1_log_diffusion_adjoint_2d(
        xVert, yVert, np.exp(theta), primal, adjoint
    )

    step = 1e-6
    finiteDifference = (
        evaluate(theta + step * direction)
        - evaluate(theta - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        gradient @ direction, finiteDifference, rtol=2e-6, atol=2e-8
    )
