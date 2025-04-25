use super::hidden_forward_parallel::hidden_forward_parallel;
use super::outer_up::outer_up;
use ndarray::{ArrayD, NdFloat};
use num_traits::Float;

/// Computes the upward pass through the layers.
///
/// Corresponds to the `up` method in the Python reference implementation.
///
/// # Arguments
/// * `layers` - A slice containing the memory tensors for each layer.
/// * `sensor_input` - The initial input tensor.
/// * `rho` - The regularization parameter.
///
/// # Returns
/// A vector containing the activation tensors for each layer after the upward pass.
///
/// # Panics
/// Panics if the `layers` slice is empty.
pub fn up<T>(
    layers: &[ArrayD<T>],    // Use slice to avoid taking ownership of the layer list
    sensor_input: ArrayD<T>, // Take ownership of the initial input
    rho: f32,
) -> Vec<ArrayD<T>>
where
    T: Clone + std::fmt::Debug + ndarray::ScalarOperand + Float + NdFloat,
{
    if layers.is_empty() {
        panic!("Layers cannot be empty for the up pass.");
    }

    // Calculate the activation for the first layer using outer_up.
    // outer_up expects Vec<ArrayD<T>> and uses the first element.
    // We clone the first layer's memory tensor to pass into outer_up.
    let first_layer_mem_vec = vec![layers[0].clone()];
    // sensor_input is moved into outer_up here.
    let initial_up = outer_up(first_layer_mem_vec, sensor_input, rho);

    // Initialize the list of upward activations.
    let mut upwards: Vec<ArrayD<T>> = vec![initial_up];

    // Iterate through the remaining layers (starting from the second layer, index 1).
    for i in 1..layers.len() {
        // Get the memory tensor for the current layer. Clone it as hidden_forward_parallel takes ownership.
        let mm = layers[i].clone();

        // Get the activation from the previous layer. Clone it as hidden_forward_parallel takes ownership.
        // Note: upwards always contains at least `i` elements at this point.
        let prev_up = upwards[i - 1].clone();

        // Compute the hidden forward pass for the current layer.
        let h_sub_l = hidden_forward_parallel(mm, prev_up, rho);

        // Add the result to the list of upward activations.
        upwards.push(h_sub_l);
    }

    // Return the collected upward activations for all layers.
    upwards
}
