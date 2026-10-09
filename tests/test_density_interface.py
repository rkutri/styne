from typing import Any

import numpy as np

import styne.statistics as statistics
from styne.backend import infer_backend
from styne.parameter import Vector
from styne.statistics.covariance import IIDCovarianceMatrix
from styne.statistics.gaussian import GaussianDensity
from styne.statistics.interface import DensityInterface


def test_density_interface_promises_only_backend_scalar_evaluation():
    annotation = DensityInterface.evaluate_log.__annotations__["return"]

    assert annotation is Any
    assert not hasattr(statistics, "DifferentiableDensity")
    assert not hasattr(statistics, "TwiceDifferentiableDensity")


def test_gaussian_log_density_returns_rank_zero_numpy_scalar():
    density = GaussianDensity(
        IIDCovarianceMatrix(2, 1.0), Vector(np.zeros(2))
    )

    value = density.evaluate_log(Vector(np.array([0.5, -1.0])))

    assert value.shape == ()
    assert infer_backend(value).name == "numpy"
