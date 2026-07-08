import pytest

@pytest.mark.skip(
    reason="SPDE and DNA-SPDE engines are experimental, excluded "
    "from the v0.2.0 frozen API.")
def test_spde_engines_todo():
    pass
