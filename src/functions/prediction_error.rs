use super::predict::predict;
use ndarray::{ArrayD, NdFloat, ScalarOperand};
use num_traits::{Float, FromPrimitive, NumCast, ToPrimitive};
use std::fmt::Debug;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/src/functions/prediction_error.rs

/// Calculates the prediction error (mean squared error) for a given input.
///
/// Corresponds to the `prediction_error` method in the Python reference implementation.
/// This function does not modify the layer weights.
///
/// # Arguments
/// * `layers` - A slice containing the memory tensors for each layer.
/// * `sensory_input` - The input tensor for prediction.
/// * `rho` - The regularization parameter used in the `up` pass within `predict`.
/// * `eps` - Epsilon value used in the `pred_down` pass within `predict`.
/// * `coeff` - Coefficient used in the `pred_down` pass within `predict`.
///
/// # Returns
/// A scalar value `T` representing the mean squared error between the prediction and the input.
///
/// # Panics
/// Panics if the underlying `predict` function panics.
pub fn prediction_error<T>(
    layers: &[ArrayD<T>],
    sensory_input: ArrayD<T>,
    rho: f32,
    eps: T,
    coeff: T,
) -> T
where
    T: NdFloat + Float + NumCast + ToPrimitive + Debug + Clone + ScalarOperand + FromPrimitive,
{
    // Clone sensory_input because predict takes ownership, but we need it later for comparison.
    let prediction = predict(layers, sensory_input.clone(), rho, eps, coeff);

    // Calculate the difference
    let diff = prediction - sensory_input;

    // Square the difference element-wise
    let squared_diff = diff.mapv(|x| x * x);

    // Calculate the mean of the squared differences
    squared_diff
        .mean()
        .expect("Cannot compute mean of empty array")
}
