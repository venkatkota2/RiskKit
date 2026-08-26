# riskcore

A dependency-free Rust library for portfolio and financial risk analytics, with a thin Python `ctypes` adapter.

The project concentrates numerical validation and risk definitions in a small compiled core. Python can orchestrate data work while Rust owns the hot, easily audited calculations. The result is a practical example of a mixed-language financial library without a heavy binding framework.

## Implemented analytics

- Historical Value at Risk and Expected Shortfall.
- Parametric Gaussian Value at Risk.
- Exponentially weighted volatility.
- Maximum drawdown from periodic returns.
- Sample covariance and portfolio volatility.
- A command-line risk report for one-column return files.
- Stable C ABI functions consumed by the Python adapter.

All functions reject empty inputs, non-finite observations, invalid confidence levels, and structurally invalid matrices.

## Build and test

```bash
cargo test
cargo build --release
cargo run -- returns.csv 0.99
```

Python adapter after a release build:

```python
from python.riskcore import RiskCore

risk = RiskCore()
returns = [0.012, -0.018, 0.004, -0.031, 0.009]
print(risk.historical_var(returns, 0.99))
print(risk.expected_shortfall(returns, 0.99))
```

## Definitions

VaR is returned as a positive loss at the requested confidence level. Historical VaR uses linearly interpolated empirical quantiles. Expected Shortfall is the mean loss at or beyond that VaR threshold. Drawdown compounds the supplied periodic returns and measures the greatest peak-to-trough decline.

## Repository layout

```text
src/lib.rs             validated analytics and C ABI
src/bin/riskcore.rs    command-line report
python/riskcore.py     zero-dependency Python adapter
tests/                 adapter integration test
```

## Scope

Risk measures are only as meaningful as the return horizon, data history, and modelling assumptions supplied to them. This project reports calculations; it does not perform data cleaning, position valuation, limit governance, or regulatory capital aggregation.

