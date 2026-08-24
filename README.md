<p align="center">
  <img src="docs/assets/styne-logo-wide.svg" alt="styne logo" width="420">
</p>

<a href="https://zenodo.org/badge/latestdoi/1278349844">
  <img src="https://zenodo.org/badge/1278349844.svg" alt="DOI">
</a>
[![DART](https://img.shields.io/badge/DART-arXiv%3A2606.27564-b31b1b)](https://arxiv.org/abs/2606.27564) [![DNA](https://img.shields.io/badge/DNA-10.1137%2F24M1715854-blue)](https://doi.org/10.1137/24M1715854)

*Pre-1.0 (`v0.3.0`)*

`styne` is a Python library for high-dimensional Bayesian inference. It keeps
models, probability measures, and samplers composable while preserving
backend-native numerical state.

## Install

Requires Python 3.10 or newer.

```bash
pip install styne
```

Optional features are installed separately:

```bash
pip install styne[jax]
pip install styne[torch]
pip install styne[plotting]
```

NumPy supports explicit gradients. JAX and PyTorch additionally provide
automatic differentiation where supported.

JAX supports compiled transformed trajectories. PyTorch samplers run eagerly,
and deterministic PyTorch operations can be compiled, but explicit
`torch.Generator` state is not supported inside a full-graph transformed
trajectory.

## Examples

The backend-neutral examples cover the main workflows:

- `examples/01_quickstart.py`: Bayesian linear regression.
- `examples/02_gp.py`: Gaussian-process representations.
- `examples/03_sglmm.py`: a spatial Poisson GLMM.

Run a fast check with:

```bash
python examples/01_quickstart.py --backend numpy --smoke
```

The NumPy/SciPy PDE example is in
`examples_numpy/04_pde_inverse_problem.py`.

## Components

- MRW, MALA, pCN, pMALA, MLDA, DART, and Gibbs samplers.
- Dense, B-spline, and DNA Gaussian-process representations.
- Linear regression, spatial GLMMs, and custom `ForwardMap` models.
- NumPy, JAX, and PyTorch numerical backends.

State is immutable and backend-native. Sampling methods accept and return
explicit random states, and parameter updates use `with_*` methods.

```text
Parameter -> ForwardMap -> Likelihood / Density -> Sampler
```

## Benchmarks

The DART benchmarks and plotting command are documented in
[`benchmarks/README.md`](benchmarks/README.md).

## Citation

Please cite the DOI shown above. For exact reproducibility, cite the archived
version used in the analysis.

## Licence

MIT. See [LICENSE](LICENSE).
