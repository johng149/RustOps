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

/// Implements a "where value" operation similar to PyTorch's y[condition] = value pattern
///
/// Returns a new array with the same values as `x`, but replaces elements with `value`
/// where `condition` is true.
///
/// # Arguments
///
/// * `condition` - A boolean array used as the condition
/// * `x` - The input array whose values will be preserved where condition is false
/// * `value` - The single value to use where condition is true
///
/// # Returns
///
/// A new ArrayD<T> with the same shape as the input arrays
pub fn where_value<T: Clone>(
    condition: &ArrayD<bool>,
    x: &ArrayD<T>,
    value: T,
) -> Result<ArrayD<T>, String> {
    // Check that condition and x have the same shape
    if condition.shape() != x.shape() {
        return Err(format!(
            "Shape mismatch: condition {:?}, x {:?}",
            condition.shape(),
            x.shape()
        ));
    }

    // Clone x to create a result array (since most values will remain the same)
    let mut result = x.clone();

    // Apply the replacement operation where condition is true
    azip!((r in &mut result, &cond in condition) {
        if cond {
            *r = value.clone();
        }
    });

    Ok(result)
}
