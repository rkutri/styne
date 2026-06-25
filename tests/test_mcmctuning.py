"""
Tests for mcmc tuning utilities.

Run with:  python -m pytest tests/test_mcmctuning.py -v
"""

import pytest

# ---------------------------------------------------------------------------
# Regression: bisection module
# ---------------------------------------------------------------------------

class TestBisectionRegression:

    def test_finds_root_of_linear_function(self):
        from styne.utility.bisection import bisection

        result = bisection(lambda x: x, 0.5, xLo=0.1, xHi=1.0, log=False)
        assert abs(result - 0.5) <= 0.05

    def test_finds_root_in_log_space(self):
        from styne.utility.bisection import bisection

        result = bisection(
            lambda x: x, 0.5,
            xLo=1e-3, xHi=10.0,
            log=True,
            tol=0.01,
        )
        assert abs(result - 0.5) / 0.5 <= 0.1

    def test_raises_when_bracket_cannot_expand(self):
        from styne.utility.bisection import bisection

        with pytest.raises(RuntimeError, match="bracket"):
            bisection(
                lambda x: 0.1, 0.9,
                xLo=0.01, xHi=0.5,
                log=False,
                xLimitLo=0.01,
                xLimitHi=0.5,
                maxExpansions=3,
            )

    def test_early_exit_at_endpoint(self):
        from styne.utility.bisection import bisection

        calls = []

        def f(x):
            calls.append(x)
            return 0.234

        result = bisection(f, 0.234, xLo=0.1, xHi=0.9, log=False, tol=0.05)
        assert len(calls) == 2
        assert result == 0.1
