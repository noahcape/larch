pub fn standard_deviation(data: &Vec<f64>) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let n = data.len() as f64;

    // Calculate the mean
    let sum: f64 = data.iter().sum();
    let mean = sum / n;

    // Calculate the variance
    let variance_sum: f64 = data
        .iter()
        .map(|&value| {
            let diff = value - mean;
            diff * diff
        })
        .sum();

    let variance = variance_sum / n;

    // Calculate the standard deviation
    Some(variance.sqrt())
}
