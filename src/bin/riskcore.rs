use riskcore::{expected_shortfall, historical_var, maximum_drawdown, parametric_var};
use std::env;
use std::fs;
use std::process;

fn read_returns(path: &str) -> Result<Vec<f64>, String> {
    let contents = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut values = Vec::new();
    for (line_number, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value = line
            .split(',')
            .next()
            .and_then(|field| field.trim().parse::<f64>().ok())
            .filter(|parsed| parsed.is_finite())
            .ok_or_else(|| format!("invalid return on line {}", line_number + 1))?;
        values.push(value);
    }
    if values.is_empty() {
        return Err("the input file contains no numeric returns".to_string());
    }
    Ok(values)
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() < 2 {
        eprintln!("usage: riskcore <returns.csv> [confidence]");
        process::exit(2);
    }
    let confidence = arguments
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.99);
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
