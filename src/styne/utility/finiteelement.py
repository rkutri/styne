import numpy as np
from scipy.sparse import diags, csc_matrix, coo_matrix

# Reference Q1 bilinear stiffness matrices on the unit square.
# Node ordering within each element:
#   N0=(1-xi)(1-eta), N1=xi(1-eta), N2=xi*eta, N3=(1-xi)*eta
# at corners (ex,ey), (ex+1,ey), (ex+1,ey+1), (ex,ey+1) respectively.
_KXI = np.array([[2, -2, -1,  1],
                 [-2,  2,  1, -1],
                 [-1,  1,  2, -2],
                 [1, -1, -2,  2]], dtype=float) / 6.0

_KETA = np.array([[2,  1, -1, -2],
                  [1,  2, -2, -1],
                  [-1, -2,  2,  1],
                  [-2, -1,  1,  2]], dtype=float) / 6.0


# ---------------------------------------------------------------------------
# 1D P1 finite elements
# ---------------------------------------------------------------------------

def p1_stiffness_1d(vertices, diffusion_coefficients) -> csc_matrix:
    """Assemble P1 stiffness matrix K for -div(A grad u) on a 1D grid.

    Parameters
    ----------
    vertices : array_like, shape (n,)
        Vertex coordinates.
    diffusion_coefficients : array_like, shape (n-1,)
        Per-element diffusion values (e.g. evaluated at element midpoints).
    """
    vertices = np.asarray(vertices, dtype=float)
    h = np.diff(vertices)
    a = np.asarray(diffusion_coefficients, dtype=float)
    vals = a / h
    n = len(vertices)
    main = np.zeros(n)
    main[:-1] += vals
    main[1:] += vals
    return diags([-vals, main, -vals], [-1, 0, 1], format='csc')


def p1_mass_lumped_1d(vertices) -> csc_matrix:
    """Lumped (row-sum) mass matrix for P1 elements on a 1D grid.

    Parameters
    ----------
    vertices : array_like, shape (n,)
        Vertex coordinates.
    """
    vertices = np.asarray(vertices, dtype=float)
    h = np.diff(vertices)
    n = len(vertices)
    mass = np.zeros(n)
    mass[:-1] += 0.5 * h
    mass[1:] += 0.5 * h
    return diags(mass, 0, format='csc')


def apply_dirichlet_1d(A, b=None):
    """Apply homogeneous Dirichlet BCs at both endpoints.

    Boundary rows of `A` are replaced by identity rows. If a right-hand
    side is given, its boundary entries are set to zero so the constrained
    system enforces u = 0 at the endpoints.

    Parameters
    ----------
    A : sparse matrix (any format)
        System matrix; converted to lil internally, returned as csc.
    b : ndarray, optional
        Right-hand side vector; a modified copy is returned. If omitted,
        only the constrained matrix is returned. In that case the caller
        must zero the boundary entries of any right-hand side used with
        the returned matrix.

    Returns
    -------
    csc_matrix, or (csc_matrix, ndarray) if `b` is given.
    """
    A = A.tolil()
    for idx in (0, A.shape[0] - 1):
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
    A = csc_matrix(A)

    if b is None:
        return A

    b = b.copy()
    b[0] = 0.0
    b[-1] = 0.0
    return A, b


def apply_neumann_1d(A, b=None):
    """Natural (homogeneous Neumann) BCs — no row modification for P1 elements.

    Parameters
    ----------
    A : sparse matrix (any format)
    b : ndarray, optional

    Returns
    -------
    csc_matrix, or (csc_matrix, ndarray) if `b` is given.
    """
    if b is None:
        return csc_matrix(A)
    return csc_matrix(A), b


# ---------------------------------------------------------------------------
# 2D Q1 bilinear finite elements
# Node index convention: node(i, j) = i * (ny + 1) + j,
# where i is the x-index (0 .. nx) and j is the y-index (0 .. ny).
# ---------------------------------------------------------------------------

def q1_stiffness_2d(xVert, yVert, diffusion_at_centers) -> csc_matrix:
    """Assemble the Q1 global stiffness matrix for -div(A grad u) on a 2D grid.

    Parameters
    ----------
    xVert : array_like, shape (nx+1,)
    yVert : array_like, shape (ny+1,)
    diffusion_at_centers : array_like, shape (nx*ny,)
        Per-element diffusion values, ordered with the x-index varying fastest
        (i.e. element eIdx = ex*ny + ey, ex in 0..nx-1, ey in 0..ny-1).
    """
    xVert = np.asarray(xVert, dtype=float)
    yVert = np.asarray(yVert, dtype=float)
    nx, ny = xVert.size - 1, yVert.size - 1
    nNodes = (nx + 1) * (ny + 1)
    nElem = nx * ny

    hx = np.diff(xVert)
    hy = np.diff(yVert)
    HX, HY = np.meshgrid(hx, hy, indexing='ij')
    HX, HY = HX.ravel(), HY.ravel()

    AEval = np.asarray(diffusion_at_centers, dtype=float).ravel()

    eIdx = np.arange(nElem)
    ex, ey = eIdx // ny, eIdx % ny

    n0 = ex * (ny + 1) + ey
    n1 = (ex + 1) * (ny + 1) + ey
    n2 = (ex + 1) * (ny + 1) + (ey + 1)
    n3 = ex * (ny + 1) + (ey + 1)
    nods = np.stack([n0, n1, n2, n3], axis=1)  # (nElem, 4)

    Ke = AEval[:, None, None] * (
        (HY / HX)[:, None, None] * _KXI[None]
        + (HX / HY)[:, None, None] * _KETA[None]
    )  # (nElem, 4, 4)

    i_idx = np.broadcast_to(nods[:, :, None], (nElem, 4, 4)).reshape(-1)
    j_idx = np.broadcast_to(nods[:, None, :], (nElem, 4, 4)).reshape(-1)

    return coo_matrix((Ke.ravel(), (i_idx, j_idx)),
                      shape=(nNodes, nNodes)).tocsc()


def q1_mass_lumped_2d(xVert, yVert) -> csc_matrix:
    """Lumped (row-sum) mass matrix for Q1 elements on a 2D grid.

    Parameters
    ----------
    xVert : array_like, shape (nx+1,)
    yVert : array_like, shape (ny+1,)
    """
    xVert = np.asarray(xVert, dtype=float)
    yVert = np.asarray(yVert, dtype=float)
    nx, ny = xVert.size - 1, yVert.size - 1

    areas = np.outer(np.diff(xVert), np.diff(yVert)) / 4.0  # (nx, ny)

    mass = np.zeros((nx + 1, ny + 1))
    mass[:-1, :-1] += areas
    mass[1:, :-1] += areas
    mass[:-1, 1:] += areas
    mass[1:, 1:] += areas

    return diags(mass.ravel(), 0, format='csc')


def apply_dirichlet_2d(A, b=None, nx=None, ny=None):
    """Apply homogeneous Dirichlet BCs on all four sides of a 2D mesh.

    Boundary rows of `A` are replaced by identity rows. If a right-hand
    side is given, its boundary entries are set to zero.

    Parameters
    ----------
    A : sparse matrix
    b : ndarray, optional
        Right-hand side vector; a modified copy is returned. If omitted,
        only the constrained matrix is returned and the caller must zero
        the boundary entries of any right-hand side used with it.
    nx, ny : int
        Number of elements in x and y directions. Required.

    Returns
    -------
    csc_matrix, or (csc_matrix, ndarray) if `b` is given.
    """
    if nx is None or ny is None:
        raise TypeError("nx and ny are required")

    A = A.tolil()

    bc_nodes = set()
    for i in range(nx + 1):
        bc_nodes.add(i * (ny + 1))
        bc_nodes.add(i * (ny + 1) + ny)
    for j in range(ny + 1):
        bc_nodes.add(j)
        bc_nodes.add(nx * (ny + 1) + j)

    for n in bc_nodes:
        A[n, :] = 0.0
        A[n, n] = 1.0
    A = csc_matrix(A)

    if b is None:
        return A

    b = b.copy()
    for n in bc_nodes:
        b[n] = 0.0

    return A, b


def apply_mixed_bc_2d(A, b_rhs, nx, ny, bc) -> tuple:
    """Apply per-dimension BCs following the DNA b ∈ {0,1}^2 convention.

    bc[j] = 0 → homogeneous Neumann on the faces perpendicular to axis j
            (no row modification; natural BC).
    bc[j] = 1 → homogeneous Dirichlet on those faces.

    Parameters
    ----------
    A : sparse matrix
    b_rhs : ndarray
    nx, ny : int
    bc : tuple[int, int]
        Per-dimension BC flags (0 = Neumann, 1 = Dirichlet).

    Returns
    -------
    (csc_matrix, ndarray)
    """
    b_rhs = b_rhs.copy()
    A = A.tolil()

    bc_nodes = set()
    if bc[0] == 1:  # Dirichlet on x = 0 and x = xVert[-1] faces
        for j in range(ny + 1):
            bc_nodes.add(j)
            bc_nodes.add(nx * (ny + 1) + j)
    if bc[1] == 1:  # Dirichlet on y = 0 and y = yVert[-1] faces
        for i in range(nx + 1):
            bc_nodes.add(i * (ny + 1))
            bc_nodes.add(i * (ny + 1) + ny)

    for n in bc_nodes:
        A[n, :] = 0.0
        A[n, n] = 1.0
        b_rhs[n] = 0.0

    return csc_matrix(A), b_rhs
