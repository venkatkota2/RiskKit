use riskcore::{expected_shortfall, historical_var, maximum_drawdown, parametric_var};
use std::env;
use std::fs;
use std::process;

fn parse_returns(contents: &str) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    for (line_number, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split(',');
        let field = fields.next().unwrap_or_default();
        if fields.next().is_some() {
            return Err(format!(
                "expected one return column on line {}",
                line_number + 1
            ));
        }
        let value = field
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|parsed| parsed.is_finite())
            .ok_or_else(|| format!("invalid return on line {}", line_number + 1))?;
        values.push(value);
    }
    if values.is_empty() {
        return Err("the input file contains no numeric returns".to_string());
    }
    Ok(values)
}

fn read_returns(path: &str) -> Result<Vec<f64>, String> {
    let contents = fs::read_to_string(path).map_err(|error| error.to_string())?;
    parse_returns(&contents)
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    if !(2..=3).contains(&arguments.len()) {
        eprintln!("usage: riskcore <returns.csv> [confidence]");
        process::exit(2);
    }
    let confidence = match arguments.get(2) {
        Some(value) => match value.parse::<f64>() {
            Ok(parsed) if parsed.is_finite() => parsed,
            _ => {
                eprintln!("error: confidence must be a finite number");
                process::exit(2);
            }
        },
        None => 0.99,
    };
    let returns = match read_returns(&arguments[1]) {
        Ok(values) => values,
        Err(error) => {
            eprintln!("error: {error}");
            process::exit(1);
        }
    };
    let result = (
        historical_var(&returns, confidence),
        expected_shortfall(&returns, confidence),
        parametric_var(&returns, confidence),
        maximum_drawdown(&returns),
    );
    match result {
        (Ok(var), Ok(es), Ok(parametric), Ok(drawdown)) => println!(
            "{{\"observations\":{},\"confidence\":{:.6},\"historical_var\":{:.10},\"expected_shortfall\":{:.10},\"parametric_var\":{:.10},\"maximum_drawdown\":{:.10}}}",
            returns.len(), confidence, var, es, parametric, drawdown
        ),
        _ => {
            eprintln!("error: invalid data or confidence level");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_returns;

    #[test]
    fn parses_a_strict_one_column_file() {
        assert_eq!(
            parse_returns("0.01\n-0.02\n\n0.03\n").unwrap(),
            [0.01, -0.02, 0.03]
        );
    }

    #[test]
    fn rejects_headers_extra_columns_and_non_finite_values() {
        assert!(parse_returns("return\n0.01\n").is_err());
        assert!(parse_returns("0.01,ignored\n").is_err());
        assert!(parse_returns("NaN\n").is_err());
    }
}
