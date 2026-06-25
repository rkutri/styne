import numpy.linalg

class NotPositiveDefinite(numpy.linalg.LinAlgError):
    """Covariance factorisation failed because the matrix is not positive definite."""
