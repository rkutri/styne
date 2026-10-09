import numpy as np
import pytest

from styne.statistics.data import Data


def test_data_preserves_pytorch_design_and_measurement():
    torch = pytest.importorskip("torch", reason="PyTorch is optional")
    design = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    measurement = torch.tensor(
        [[2.0], [3.0]], dtype=torch.float32, requires_grad=True
    )

    data = Data(1, design)
    data.measurement = measurement

    assert data.design is design
    assert data.measurement is measurement
    assert data.measurement.dtype == torch.float32
    data.measurement.sum().backward()
    torch.testing.assert_close(measurement.grad, torch.ones_like(measurement))


def test_data_preserves_jax_design_and_measurement():
    jnp = pytest.importorskip("jax.numpy", reason="JAX is optional")
    design = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
    measurement = jnp.array([[2.0], [3.0]], dtype=jnp.float32)

    data = Data(1, design)
    data.measurement = measurement

    assert type(data.design) is type(design)
    assert type(data.measurement) is type(measurement)
    assert data.design.dtype == design.dtype
    assert data.measurement.dtype == measurement.dtype
    np.testing.assert_allclose(data.measurement, measurement)
