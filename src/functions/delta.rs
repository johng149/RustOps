use super::delta_outer::delta_outer;
use super::mem_delta::mem_delta;
use ndarray::{ArrayD, NdFloat};

/// Calculates the deltas for all layers.
/// This function does not mutate its inputs.
///
/// # Arguments
/// * `downwards` - Vector of downward activation tensors, ordered from outer to inner layer.
/// * `layers_mm` - Vector of memory matrices for each layer, ordered from outer to inner layer.
/// * `sensory_input` - The sensory input tensor (used for the outermost layer).
///
/// # Returns
/// * A vector of delta tensors for each layer, ordered from outer to inner layer.
pub fn delta<T>(
    downwards: &Vec<ArrayD<T>>,
    layers_mm: &Vec<ArrayD<T>>,
    sensory_input: ArrayD<T>,
) -> Vec<ArrayD<T>>
where
    T: NdFloat,
{
    if downwards.is_empty() || layers_mm.is_empty() {
        panic!("Downwards and layers_mm vectors must not be empty.");
    }
    if downwards.len() != layers_mm.len() {
        panic!("Number of downward activations must match the number of layer memory matrices.");
    }

    // Calculate delta for the outermost layer (index 0)
    let outer_delta = delta_outer(downwards, layers_mm, &sensory_input);
    let mut deltas = vec![outer_delta];

    let num_downwards = downwards.len();

    // Calculate deltas for the inner layers (index 1 to N-1)
    // Python: for i in range(1, len(downwards)):
    // Rust equivalent iterates i from 1 up to num_downwards - 1
    for i in 1..num_downwards {
        // Python: child_h_sub_l_star = downwards[len(downwards) - i]
        // Corresponds to the downward activation *from* the layer *above* the current layer 'i'
        // In our Rust vectors (outer to inner), this is downwards[i-1]
        // But the Python code indexes from the *end* of the downwards list.
        // downwards[-1] is the innermost activation.
        // downwards[len(downwards) - i] maps to downwards[num_downwards - i] in Rust
        let child_h_sub_l_star = &downwards[num_downwards - i];

        // Python: h_sub_l_star = downwards[len(downwards) - i - 1]
        // Corresponds to the downward activation *for* the current layer 'i'
        // In our Rust vectors (outer to inner), this is downwards[i]
        // Python maps this to downwards[num_downwards - i - 1] in Rust
        let h_sub_l_star = &downwards[num_downwards - i - 1];

        // Python: mm = self.layers[i]
        // Corresponds to the memory matrix for the current layer 'i'
        let mm = &layers_mm[i];

        // Calculate delta using mem_delta
        let layer_delta = mem_delta(h_sub_l_star, mm, child_h_sub_l_star);
        deltas.push(layer_delta);
    }

    // The Python code seems to return deltas ordered [outer, inner1, inner2, ...]
    // Our loop calculates them in the order corresponding to layers[1], layers[2], ...
    // which matches the desired output order when prepended with outer_delta.
    deltas
}
