//! Auditable portfolio-risk analytics with no third-party dependencies.

use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
pub enum RiskError {
    EmptyInput,
    NonFiniteInput,
    InvalidConfidence,
    InvalidDecay,
    InvalidReturn,
    LengthMismatch,
    InvalidCovarianceMatrix,
}

fn validate_returns(values: &[f64]) -> Result<(), RiskError> {
    if values.is_empty() {
        return Err(RiskError::EmptyInput);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(RiskError::NonFiniteInput);
    }
    Ok(())
}

fn validate_confidence(confidence: f64) -> Result<(), RiskError> {
    if !confidence.is_finite() || !(0.5..1.0).contains(&confidence) {
        return Err(RiskError::InvalidConfidence);
    }
    Ok(())
}

fn quantile(mut values: Vec<f64>, probability: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    if values.len() == 1 {
        return values[0];
    }
    let position = probability * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    values[lower] * (1.0 - weight) + values[upper] * weight
}

pub fn historical_var(returns: &[f64], confidence: f64) -> Result<f64, RiskError> {
    validate_returns(returns)?;
    validate_confidence(confidence)?;
    let losses = returns.iter().map(|value| -value).collect();
    Ok(quantile(losses, confidence).max(0.0))
}

pub fn expected_shortfall(returns: &[f64], confidence: f64) -> Result<f64, RiskError> {
    validate_returns(returns)?;
    validate_confidence(confidence)?;
    let mut losses: Vec<f64> = returns.iter().map(|value| -value).collect();
    losses.sort_by(|left, right| right.partial_cmp(left).unwrap_or(Ordering::Equal));

    let tail_observations = (1.0 - confidence) * losses.len() as f64;
    let whole_observations = tail_observations.floor() as usize;
    let boundary_fraction = tail_observations - whole_observations as f64;
    let mut tail_total = losses[..whole_observations].iter().sum::<f64>();
    if boundary_fraction > 0.0 {
        tail_total += boundary_fraction * losses[whole_observations];
    }
    Ok((tail_total / tail_observations).max(0.0))
}

pub fn mean(returns: &[f64]) -> Result<f64, RiskError> {
    validate_returns(returns)?;
    Ok(returns.iter().sum::<f64>() / returns.len() as f64)
}

pub fn sample_volatility(returns: &[f64]) -> Result<f64, RiskError> {
    validate_returns(returns)?;
    if returns.len() < 2 {
        return Ok(0.0);
    }
    let average = mean(returns)?;
    let variance = returns
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;
    Ok(variance.sqrt())
}

pub fn ewma_volatility(returns: &[f64], decay: f64) -> Result<f64, RiskError> {
    validate_returns(returns)?;
    if !decay.is_finite() || !(0.0..1.0).contains(&decay) {
        return Err(RiskError::InvalidDecay);
    }
    let mut variance = returns[0] * returns[0];
    for value in &returns[1..] {
        variance = decay * variance + (1.0 - decay) * value * value;
    }
    Ok(variance.sqrt())
}

pub fn maximum_drawdown(returns: &[f64]) -> Result<f64, RiskError> {
    validate_returns(returns)?;
    let mut wealth: f64 = 1.0;
    let mut peak: f64 = 1.0;
    let mut maximum: f64 = 0.0;
    for value in returns {
        if *value <= -1.0 {
            return Err(RiskError::InvalidReturn);
        }
        wealth *= 1.0 + value;
        peak = peak.max(wealth);
        let drawdown = 1.0 - wealth / peak;
        if drawdown > maximum {
            maximum = drawdown;
        }
    }
    Ok(maximum)
}

// Peter J. Acklam's inverse-normal approximation.
fn inverse_normal_cdf(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.02425;
    const HIGH: f64 = 1.0 - LOW;

    if probability < LOW {
        let q = (-2.0 * probability.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if probability <= HIGH {
        let q = probability - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - probability).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

pub fn parametric_var(returns: &[f64], confidence: f64) -> Result<f64, RiskError> {
    validate_confidence(confidence)?;
    let average = mean(returns)?;
    let volatility = sample_volatility(returns)?;
    Ok((-average + inverse_normal_cdf(confidence) * volatility).max(0.0))
}

pub fn covariance(left: &[f64], right: &[f64]) -> Result<f64, RiskError> {
    validate_returns(left)?;
    validate_returns(right)?;
    if left.len() != right.len() {
        return Err(RiskError::LengthMismatch);
    }
    if left.len() < 2 {
        return Ok(0.0);
    }
    let left_mean = mean(left)?;
    let right_mean = mean(right)?;
    Ok(left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - left_mean) * (b - right_mean))
        .sum::<f64>()
        / (left.len() - 1) as f64)
}

pub fn portfolio_volatility(weights: &[f64], covariance_matrix: &[f64]) -> Result<f64, RiskError> {
    if weights.is_empty() {
        return Err(RiskError::EmptyInput);
    }
    if covariance_matrix.len() != weights.len() * weights.len() {
        return Err(RiskError::LengthMismatch);
    }
    if weights
        .iter()
        .chain(covariance_matrix)
        .any(|value| !value.is_finite())
    {
        return Err(RiskError::NonFiniteInput);
    }
    let size = weights.len();
    validate_covariance_matrix(covariance_matrix, size)?;
    let mut variance = 0.0;
    for (row, row_weight) in weights.iter().enumerate() {
        for (column, column_weight) in weights.iter().enumerate() {
            variance += row_weight * covariance_matrix[row * size + column] * column_weight;
        }
    }
    let scale = covariance_matrix
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    let variance_tolerance = 1e-12 * scale * size as f64;
    if variance < -variance_tolerance {
        return Err(RiskError::InvalidCovarianceMatrix);
    }
    Ok(variance.max(0.0).sqrt())
}

fn validate_covariance_matrix(matrix: &[f64], size: usize) -> Result<(), RiskError> {
    let scale = matrix
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    let tolerance = 1e-12 * scale * size as f64;

    for row in 0..size {
        if matrix[row * size + row] < -tolerance {
            return Err(RiskError::InvalidCovarianceMatrix);
        }
        for column in 0..row {
            if (matrix[row * size + column] - matrix[column * size + row]).abs() > tolerance {
                return Err(RiskError::InvalidCovarianceMatrix);
            }
        }
    }

    // Tolerance-aware LDL^T factorization. A zero pivot is valid for a
    // semidefinite matrix only when the corresponding residual column is zero.
    let mut lower = vec![0.0; size * size];
    let mut diagonal = vec![0.0; size];
    for row in 0..size {
        for column in 0..row {
            let mut residual = matrix[row * size + column];
            for previous in 0..column {
                residual -= lower[row * size + previous]
                    * diagonal[previous]
                    * lower[column * size + previous];
            }
            if diagonal[column] > tolerance {
                lower[row * size + column] = residual / diagonal[column];
            } else if residual.abs() > tolerance {
                return Err(RiskError::InvalidCovarianceMatrix);
            }
        }

        let mut pivot = matrix[row * size + row];
        for previous in 0..row {
            let entry = lower[row * size + previous];
            pivot -= entry * entry * diagonal[previous];
        }
        if pivot < -tolerance {
            return Err(RiskError::InvalidCovarianceMatrix);
        }
        diagonal[row] = if pivot.abs() <= tolerance { 0.0 } else { pivot };
        lower[row * size + row] = 1.0;
    }
    Ok(())
}

unsafe fn slice_from_ffi<'a>(values: *const f64, length: usize) -> Option<&'a [f64]> {
    if values.is_null() || length == 0 {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(values, length) })
    }
}

/// Compute historical VaR through the C ABI.
///
/// # Safety
/// `values` must point to `length` readable, properly aligned `f64` values for the duration of
/// the call. The memory must not be mutated concurrently.
#[no_mangle]
pub unsafe extern "C" fn riskcore_historical_var(
    values: *const f64,
    length: usize,
    confidence: f64,
) -> f64 {
    unsafe { slice_from_ffi(values, length) }
        .and_then(|slice| historical_var(slice, confidence).ok())
        .unwrap_or(f64::NAN)
}

/// Compute Expected Shortfall through the C ABI.
///
/// # Safety
/// `values` must point to `length` readable, properly aligned `f64` values for the duration of
/// the call. The memory must not be mutated concurrently.
#[no_mangle]
pub unsafe extern "C" fn riskcore_expected_shortfall(
    values: *const f64,
    length: usize,
    confidence: f64,
) -> f64 {
    unsafe { slice_from_ffi(values, length) }
        .and_then(|slice| expected_shortfall(slice, confidence).ok())
        .unwrap_or(f64::NAN)
}

/// Compute maximum drawdown through the C ABI.
///
/// # Safety
/// `values` must point to `length` readable, properly aligned `f64` values for the duration of
/// the call. The memory must not be mutated concurrently.
#[no_mangle]
pub unsafe extern "C" fn riskcore_maximum_drawdown(values: *const f64, length: usize) -> f64 {
    unsafe { slice_from_ffi(values, length) }
        .and_then(|slice| maximum_drawdown(slice).ok())
        .unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RETURNS: [f64; 10] = [
        0.012, -0.018, 0.004, -0.031, 0.009, -0.006, 0.015, -0.024, 0.011, -0.008,
    ];

    #[test]
    fn expected_shortfall_is_at_least_var() {
        let var = historical_var(&RETURNS, 0.95).unwrap();
        let shortfall = expected_shortfall(&RETURNS, 0.95).unwrap();
        assert!(shortfall >= var);
        assert!(var > 0.0);
    }

    #[test]
    fn risk_statistics_match_known_examples() {
        assert!((mean(&[1.0, 2.0, 3.0]).unwrap() - 2.0).abs() < 1e-12);
        assert!((sample_volatility(&[1.0, 2.0, 3.0]).unwrap() - 1.0).abs() < 1e-12);
        assert!((covariance(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn portfolio_volatility_uses_full_covariance_matrix() {
        let weights = [0.6, 0.4];
        let matrix = [0.04, 0.006, 0.006, 0.01];
        let result = portfolio_volatility(&weights, &matrix).unwrap();
        assert!((result - 0.1374045).abs() < 1e-6);
    }

    #[test]
    fn portfolio_volatility_accepts_singular_psd_matrix() {
        let weights = [0.5, 0.5];
        let matrix = [0.04, 0.04, 0.04, 0.04];
        assert!((portfolio_volatility(&weights, &matrix).unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn invalid_covariance_matrices_are_rejected() {
        let weights = [0.5, 0.5];
        assert_eq!(
            portfolio_volatility(&weights, &[0.04, 0.01, 0.02, 0.03]),
            Err(RiskError::InvalidCovarianceMatrix)
        );
        assert_eq!(
            portfolio_volatility(&weights, &[0.04, 0.06, 0.06, 0.04]),
            Err(RiskError::InvalidCovarianceMatrix)
        );
        assert_eq!(
            portfolio_volatility(&weights, &[-0.01, 0.0, 0.0, 0.02]),
            Err(RiskError::InvalidCovarianceMatrix)
        );
        assert_eq!(
            portfolio_volatility(&weights, &[0.04, f64::NAN, f64::NAN, 0.03]),
            Err(RiskError::NonFiniteInput)
        );
        assert_eq!(
            portfolio_volatility(&weights, &[0.04, 0.01]),
            Err(RiskError::LengthMismatch)
        );
    }

    #[test]
    fn expected_shortfall_uses_fractional_empirical_tail() {
        // At 80% confidence, exactly two of ten observations belong in the tail.
        let returns = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.10, -0.20];
        assert!((expected_shortfall(&returns, 0.80).unwrap() - 0.15).abs() < 1e-12);

        // One positive loss fills half the tail; the zero boundary fills the rest.
        let sparse_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.10];
        assert!((expected_shortfall(&sparse_loss, 0.80).unwrap() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert_eq!(historical_var(&[], 0.95), Err(RiskError::EmptyInput));
        assert_eq!(
            historical_var(&RETURNS, 1.0),
            Err(RiskError::InvalidConfidence)
        );
        assert_eq!(maximum_drawdown(&[-1.0]), Err(RiskError::InvalidReturn));
    }
}
