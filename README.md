# RiskKit

[![CI](https://github.com/venkatkota2/RiskKit/actions/workflows/ci.yml/badge.svg)](https://github.com/venkatkota2/RiskKit/actions/workflows/ci.yml)

A small, auditable Rust financial-risk core with a thin Python `ctypes` interoperability layer. The compiled core and command-line executable are named `riskcore`.

The project concentrates numerical validation and risk definitions in a small compiled core. Python can orchestrate data work while Rust owns the hot, easily audited calculations. The result is a practical example of a mixed-language financial library without a heavy binding framework.

## Implemented analytics

- Historical Value at Risk and Expected Shortfall.
- Parametric Gaussian Value at Risk.
- Exponentially weighted volatility.
- Maximum drawdown from periodic returns.
- Sample covariance and portfolio volatility.
- A command-line risk report for one-column return files.
- Stable C ABI functions consumed by the Python adapter.

All functions reject empty inputs, non-finite observations, invalid confidence levels, and structurally invalid matrices. Portfolio volatility validates symmetry, diagonal values, and positive-semidefinite structure with a tolerance-aware dependency-free `LDL^T` factorization; it does not silently turn a materially negative variance into zero.

## Build and test

```bash
cargo test
cargo build --release
cargo run -- returns.csv 0.99
```

The CLI accepts a headerless, one-column numeric file. Extra columns,
non-finite observations, malformed confidence arguments, and unexpected
positional arguments fail explicitly instead of being ignored or defaulted.

Python adapter after a debug or release build from the source tree:

```python
from python.riskcore import RiskCore

risk = RiskCore()
returns = [0.012, -0.018, 0.004, -0.031, 0.009]
print(risk.historical_var(returns, 0.99))
print(risk.expected_shortfall(returns, 0.99))
```

## Definitions

VaR is returned as a positive loss at the requested confidence level. Historical VaR uses linearly interpolated empirical quantiles. Expected Shortfall averages exactly the worst `1 - confidence` empirical mass, including a fractional boundary observation when needed. Drawdown compounds the supplied periodic returns and measures the greatest peak-to-trough decline.

The Python adapter discovers `.dll`, `.dylib`, and `.so` builds under the local
`target` directory, or accepts an explicit `RISKCORE_LIBRARY` path. It is a
source-tree adapter; this repository does not claim to publish binary wheels.

## Repository layout

```text
src/lib.rs             validated analytics and C ABI
src/bin/riskcore.rs    command-line report
python/riskcore.py     zero-dependency Python adapter
tests/                 adapter integration test
```

## Scope

Risk measures are only as meaningful as the return horizon, data history, and modelling assumptions supplied to them. This project is an auditable calculation core, not institutional risk-management software: it does not perform data cleaning, position valuation, limit governance, or regulatory capital aggregation.
