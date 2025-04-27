use super::unsqueeze::unsqueeze;
use ndarray::{ArrayD, NdFloat}; // Assuming unsqueeze is in the same 'functions' module

/// Calculates the updated memory matrix for the outermost layer based on deltas and counts.
/// This function does not mutate its inputs.
///
/// Corresponds to the Python logic:
/// ```python
/// outer_delta = deltas[0]
/// outer_mm = self.layers[0] # In Rust, this is layers_mm[0]
/// outer_counts = self.layer_counts[0]
/// delta = outer_delta / (outer_counts).unsqueeze(-1)
/// updated_outer_mm = outer_mm + delta # This is the value returned by the Rust function
/// ```
///
/// # Arguments
/// * `deltas` - Vector of delta tensors for each layer, ordered from outer to inner layer.
///            Must contain at least one element.
/// * `layers_mm` - Vector of memory matrices for each layer, ordered from outer to inner layer.
///               Must contain at least one element.
/// * `layer_counts` - Vector of count tensors for each layer, ordered from outer to inner layer.
///                  Must contain at least one element.
///
/// # Returns
/// An `ArrayD<T>` representing the updated memory matrix for the outermost layer (`layers_mm[0]`).
///
/// # Panics
/// Panics if any of the input vectors are empty.
/// Panics if division by zero occurs in `layer_counts`.
/// Panics if shapes are incompatible for broadcasting during division or addition.
pub fn optim_outer<T>(
    deltas: &Vec<ArrayD<T>>,
    layers_mm: &Vec<ArrayD<T>>,
    layer_counts: &Vec<ArrayD<T>>,
) -> ArrayD<T>
where
    T: NdFloat + Copy, // Copy might be needed by NdFloat ops
{
    if deltas.is_empty() || layers_mm.is_empty() || layer_counts.is_empty() {
        // Consider returning a Result for more robust error handling
        panic!("Input vectors (deltas, layers_mm, layer_counts) must not be empty.");
    }

    let outer_delta = &deltas[0];
    let outer_mm = &layers_mm[0];
    let outer_counts = &layer_counts[0];

    // Unsqueeze counts at the last dimension (equivalent to Python's unsqueeze(-1))
    // The dimension index should be equal to the number of dimensions to append at the end.
    let dim_to_unsqueeze = outer_counts.ndim();
    let unsqueezed_counts = unsqueeze(outer_counts, dim_to_unsqueeze);

    // Calculate the update term: update = outer_delta / unsqueezed_counts
    // This performs element-wise division with broadcasting.
    // Ensure T implements Div. NdFloat usually covers this.
    // This will panic if any element in unsqueezed_counts is zero.
    let update_term = outer_delta / &unsqueezed_counts;

    // Calculate the new outermost layer's memory matrix: new_outer_mm = outer_mm + update_term
    // This performs element-wise addition with broadcasting.
    // Requires T to implement Add. NdFloat implies Add.
    let new_outer_mm = outer_mm + &update_term;

    new_outer_mm
}
