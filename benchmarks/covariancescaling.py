"""Compare covariance rescaling with a Git baseline on NumPy.

Run from a checkout with styne installed:
    python benchmarks/covariancescaling.py --baseline b2e0cad

The baseline covariance implementation uses the current NumPy backend. Timings
exclude the initial factorisation and report the median of repeated batches.
"""
import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
from timeit import repeat
from types import ModuleType

import numpy as np

from styne.statistics.covariance import DenseCovarianceMatrix


def baseline_covariance(revision):
    root = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root),
         "show", f"{revision}:src/styne/statistics/covariance.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    module = ModuleType("baselinecovariance")
    exec(compile(source, f"{revision}:covariance.py", "exec"), vars(module))
    return module.DenseCovarianceMatrix


def measure(covarianceType, matrix, count, repeats):
    covariance = covarianceType(matrix)
    seconds = median(repeat(
        lambda: covariance.with_scaling(0.7), number=count, repeat=repeats,
    ))
    backend = covariance._backend
    cholesky = backend.cholesky
    calls = []

    def counted(value):
        calls.append(1)
        return cholesky(value)

    backend.cholesky = counted
    try:
        for iteration in range(count):
            scaled = covariance.with_scaling(0.7)
    finally:
        backend.cholesky = cholesky
    vector = np.ones(matrix.shape[-1])
    np.testing.assert_allclose(scaled.apply(vector), 0.7 * matrix @ vector)
    return {
        "microsecondsPerRescale": seconds * 1e6 / count,
        "choleskyCalls": len(calls),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="b2e0cad")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if min(arguments.dimension, arguments.count, arguments.repeats) < 1:
        parser.error("dimension, count and repeats must be positive")
    rng = np.random.default_rng(83)
    root = rng.normal(size=(arguments.dimension, arguments.dimension))
    matrix = root @ root.T + np.eye(arguments.dimension)
    result = {
        "baseline": arguments.baseline,
        "dimension": arguments.dimension,
        "count": arguments.count,
        "repeats": arguments.repeats,
        "measurements": {
            name: measure(covarianceType, matrix, arguments.count, arguments.repeats)
            for name, covarianceType in (
                ("baseline", baseline_covariance(arguments.baseline)),
                ("updated", DenseCovarianceMatrix),
            )
        },
    }
    print(json.dumps(result, indent=2))
