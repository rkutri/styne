import pytest
import numpy as np

from styne.mcmc.chain import Chain


def test_chain_preserves_numpy_array_identity():
    sample = np.array([1.0], dtype=np.float64)
    chain = Chain()

    chain.append(sample)

    assert chain.trajectory[0] is sample
    assert chain.trajectory[0].dtype == np.float64


def test_chain_preserves_pytorch_tensor_identity_and_device():
    torch = pytest.importorskip("torch")
    sample = torch.tensor([1.0], dtype=torch.float64)
    chain = Chain()

    chain.append(sample)

    assert chain.trajectory[0] is sample
    assert chain.trajectory[0].dtype == torch.float64
    assert chain.trajectory[0].device == sample.device


def test_chain_preserves_jax_array_dtype_and_device():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    sample = jnp.array([1.0], dtype=jnp.float32)
    chain = Chain()

    chain.append(sample)

    stored = chain.trajectory[0]
    assert isinstance(stored, jax.Array)
    assert stored.dtype == sample.dtype
    assert stored.device == sample.device
