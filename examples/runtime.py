"""Command-line and presentation helpers for maintained examples."""

import argparse

import numpy as np

from styne.backend import get_backend


def parse_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--backend",
        choices=("numpy", "pytorch", "jax"),
        default="numpy",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def configure_backend(backendName, seed=2026):
    backend = get_backend(backendName)
    print(f"backend={backend.name}")
    return backend, backend.random_state(seed)


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)
