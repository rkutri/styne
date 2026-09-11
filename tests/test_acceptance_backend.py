import numpy as np
import pytest

from styne.backend import get_backend
from styne.mcmc.acceptance import BarkerAcceptance, StandardAcceptance


@pytest.mark.parametrize(
    ("acceptance", "expected"),
    [
        (
            StandardAcceptance(),
            np.array([-np.inf, -2., 0., 0., -np.inf]),
        ),
        (
            BarkerAcceptance(),
            np.array([
                -np.inf,
                -np.logaddexp(0., 2.),
                -np.log(2.),
                -np.logaddexp(0., -2.),
                -np.inf,
            ]),
        ),
    ],
)
def test_acceptance_preserves_numpy_arrays(acceptance, expected):
    logMHRatio = np.array([-np.inf, -2., 0., 2., np.nan])

    actual = acceptance.log_probability(logMHRatio)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected, equal_nan=False)


@pytest.mark.parametrize(
    "acceptance", [StandardAcceptance(), BarkerAcceptance()]
)
def test_acceptance_compiles_with_jax(acceptance):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    backend = get_backend("jax")

    logMHRatio = jnp.array([-2., 0., 2.])
    actual = jax.jit(acceptance.log_probability)(logMHRatio)
    expected = acceptance.log_probability(np.array([-2., 0., 2.]))

    assert backend.is_array(actual)
    np.testing.assert_allclose(np.asarray(actual), expected)


@pytest.mark.parametrize(
    "acceptance", [StandardAcceptance(), BarkerAcceptance()]
)
def test_acceptance_compiles_with_pytorch(acceptance):
    torch = pytest.importorskip("torch")
    get_backend("pytorch")

    logMHRatio = torch.tensor([-2., 0., 2.])
    actual = torch.compile(
        acceptance.log_probability, backend="eager", fullgraph=True
    )(logMHRatio)
    expected = acceptance.log_probability(np.array([-2., 0., 2.]))

    assert isinstance(actual, torch.Tensor)
    np.testing.assert_allclose(actual.detach().numpy(), expected)
