import pytest

from python.riskcore import RiskCore


def test_python_adapter_calls_rust_core():
    core = RiskCore()
    returns = [0.012, -0.018, 0.004, -0.031, 0.009, -0.006]

    var = core.historical_var(returns, 0.95)
    expected_shortfall = core.expected_shortfall(returns, 0.95)
    drawdown = core.maximum_drawdown(returns)

    assert expected_shortfall >= var > 0
    assert 0 < drawdown < 1


def test_python_adapter_translates_ffi_failures():
    core = RiskCore()

    with pytest.raises(ValueError, match="finite observations"):
        core.historical_var([0.01, float("nan")])
    with pytest.raises(ValueError, match="rejected"):
        core.historical_var([0.01, -0.02], 1.0)
    with pytest.raises(ValueError, match="rejected"):
        core.maximum_drawdown([0.01, -1.0])
