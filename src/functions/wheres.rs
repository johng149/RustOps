use ndarray::{ArrayD, azip};

/// Implements a where operation similar to PyTorch's torch.where()
///
/// Returns a new array with elements chosen from `x` or `y` depending on `condition`.
///
/// # Arguments
///
/// * `condition` - A boolean array used as the condition
/// * `x` - Values to use where condition is true
/// * `y` - Values to use where condition is false
///
/// # Returns
///
/// A new ArrayD<f32> with the same shape as the input arrays
pub fn where_op(
    condition: &ArrayD<bool>,
    x: &ArrayD<f32>,
    y: &ArrayD<f32>,
) -> Result<ArrayD<f32>, String> {
    // Check that all arrays have the same shape
    if condition.shape() != x.shape() || condition.shape() != y.shape() {
        return Err(format!(
            "Shape mismatch: condition {:?}, x {:?}, y {:?}",
            condition.shape(),
            x.shape(),
            y.shape()
        ));
    }

    // Create a new array to store the result
    let mut result = ArrayD::zeros(condition.shape());

    // Apply the where condition using azip! macro
    azip!((r in &mut result, &cond in condition, &x_val in x, &y_val in y) {
        *r = if cond { x_val } else { y_val };
    });

    Ok(result)
}
