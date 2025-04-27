use super::pred::pred;
use super::pred_down::pred_down;
use super::up::up;
use ndarray::{ArrayD, NdFloat, ScalarOperand};
use num_traits::{Float, NumCast, ToPrimitive};
use std::fmt::Debug;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/src/functions/predict.rs

/// Performs the prediction process: upward pass, predictive downward pass, and final prediction.
///
/// Corresponds to the `predict` method in the Python reference implementation.
/// This function does not modify the layer weights.
///
/// # Arguments
/// * `layers` - A slice containing the memory tensors for each layer.
/// * `sensory_input` - The input tensor for prediction.
/// * `rho` - The regularization parameter used in the `up` pass.
/// * `eps` - Epsilon value used in the `pred_down` pass (`argmaxi`).
/// * `coeff` - Coefficient used in the `pred_down` pass (`down_prop_parallel`).
///
/// # Returns
/// An `ArrayD<T>` containing the final prediction based on the input.
///
/// # Panics
/// Panics if:
/// * `layers` is empty.
/// * The number of upward activations and layers mismatch during `pred_down`.
/// * Any underlying function (`up`, `pred_down`, `pred`) panics.
pub fn predict<T>(
    layers: &[ArrayD<T>],
    sensory_input: ArrayD<T>,
    rho: f32, // Required by `up`
    eps: T,
    coeff: T,
) -> ArrayD<T>
where
    T: NdFloat + Float + NumCast + ToPrimitive + Debug + Clone + ScalarOperand,
{
    if layers.is_empty() {
        panic!("Layers cannot be empty for prediction.");
    }

    // 1. Upward pass
    // `up` takes ownership of sensory_input.
    let upwards = up(layers, sensory_input, rho);

    // 2. Predictive downward pass
    // `pred_down` takes upwards and layers by reference/slice.
    // Note: pred_down expects layers.len() == upwards.len().
    // The `up` function ensures upwards.len() == layers.len().
    let downwards = pred_down(&upwards, layers, eps, coeff);

    // 3. Final prediction
    // `pred` uses the last element of the downward pass (prediction at the input level)
    // and the memory matrix of the first layer.
    let final_down_prop = downwards
        .last()
        .expect("Downwards vector should not be empty after pred_down");
    let first_layer_mem = &layers[0];

    pred(final_down_prop, first_layer_mem)
}
