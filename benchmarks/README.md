# Benchmarks

- `dartlogistic.py` compares pilot-tuned DART, MALA, and MLDA.
- `dartlocalisation.py` compares localised and unlocalised DART.
- `dartvisualisation.py` plots the latest production CSVs.
- `covariancepreparation.py` compares reconstruction from a raw factor with
  retaining a covariance object during proposal preparation. Run it with
  `python benchmarks/covariancepreparation.py`. It reports microseconds per
  preparation at three dimensions; these are not whole-chain timings.

The localisation benchmark asks whether keeping DART proposals closer to the
current state helps when its Gaussian approximation has the wrong centre,
wrong spread, or both. A ratio of one means equal sampling efficiency.
It compares matched localised and unlocalised chains on 20 synthetic logistic
posteriors at dimensions 4-16, measures worst-direction ESS per target
evaluation, and fits the efficiency ratio as a power law in dimension.

Smoke tests use NumPy by default:

```bash
python benchmarks/dartlogistic.py
python benchmarks/dartlocalisation.py
```

Run production serially:

```bash
uv run --isolated --extra jax python benchmarks/dartlogistic.py --production
uv run --isolated --extra jax python benchmarks/dartlocalisation.py --production
```

Each run writes one CSV under `benchmarks/results/`. If interrupted, repeat
the same command with `--resume`.

Plot the latest available production results:

```bash
uv run --isolated --extra plotting python benchmarks/dartvisualisation.py
```

The figure is written to `benchmarks/results/dartbenchmarks.pdf`.
