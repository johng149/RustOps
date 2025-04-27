use super::optim_outer::optim_outer;
use super::unsqueeze::unsqueeze;
use ndarray::{ArrayD, IxDyn, NdFloat};
use num_traits::FromPrimitive; // Required for T::from(usize)
use std::vec::Vec;

/// Performs the optimization step for all layers and updates iteration state.
///
/// This function first updates the outermost layer using `optim_outer`,
/// then iterates through the inner layers, applying updates based on deltas and counts
/// with specific dimension unsqueezing. Finally, it updates the iteration count
/// and growth threshold. This function does not mutate its inputs.
///
/// Corresponds to the Python logic:
/// ```python
/// def optim(self, deltas):
///     self.optim_outer(deltas) # Handled by calling optim_outer
///     for i in range(1, len(deltas)):
///       delta = deltas[i]
///       mm = self.layers[i]
///       count = self.layer_counts[i]
///       # Unsqueeze count at last dim (-1) and third-to-last dim (-3)
///       delta = delta / (count).unsqueeze(-1).unsqueeze(-3)
///       self.layers[i] = mm + delta # Store updated layer
///     # Update iteration state
///     self.t += 1
///     self.growth_threshold = self.a / (self.t + self.a)
/// ```
///
/// # Arguments
/// * `deltas` - Vector of delta tensors for each layer, ordered from outer to inner.
/// * `layers_mm` - Vector of memory matrices for each layer, ordered from outer to inner.
/// * `layer_counts` - Vector of count tensors for each layer, ordered from outer to inner.
/// * `t` - The current iteration count.
/// * `a` - A parameter used for calculating the growth threshold.
///
/// # Returns
/// A tuple containing:
/// * `Vec<ArrayD<T>>` - The updated memory matrices for all layers.
/// * `usize` - The incremented iteration count (`t + 1`).
/// * `T` - The newly calculated growth threshold.
///
/// # Panics
/// Panics if input vectors have different non-zero lengths.
/// Panics if any input vector is empty.
/// Panics if division by zero occurs in `layer_counts`.
/// Panics if shapes are incompatible for broadcasting.
/// Panics if `layer_counts[i]` for `i > 0` has fewer than 2 dimensions (required for `unsqueeze(-3)`).
/// Panics if `T::from<usize>` fails (should not happen for standard float types).
pub fn optim<T>(
    deltas: &Vec<ArrayD<T>>,
    layers_mm: &Vec<ArrayD<T>>,
    layer_counts: &Vec<ArrayD<T>>,
    t: usize,
    a: T,
) -> (Vec<ArrayD<T>>, usize, T)
where
    T: NdFloat + Copy + FromPrimitive, // FromPrimitive needed for T::from(new_t)
{
    let n_layers = deltas.len();
    if n_layers == 0 || layers_mm.len() != n_layers || layer_counts.len() != n_layers {
        panic!(
            "Input vectors (deltas, layers_mm, layer_counts) must be non-empty and have the same length."
        );
    }

    let mut updated_layers_mm = Vec::with_capacity(n_layers);

    // 1. Handle the outermost layer (index 0) using optim_outer
    let updated_outer_mm = optim_outer(deltas, layers_mm, layer_counts);
    updated_layers_mm.push(updated_outer_mm);

    // 2. Iterate through the inner layers (index 1 to n_layers - 1)
    for i in 1..n_layers {
        let delta_i = &deltas[i];
        let mm_i = &layers_mm[i];
        let count_i = &layer_counts[i];

        // Unsqueeze count at the last dimension (-1)
        let count_ndim = count_i.ndim();
        let unsqueezed_count_1 = unsqueeze(count_i, count_ndim); // dim = ndim inserts at the end

        // Unsqueeze the result at the third-to-last dimension (-3)
        let temp_ndim = unsqueezed_count_1.ndim();

        // The index for -3 is ndim - 2
        let dim_neg_3 = temp_ndim - 2;
        let doubly_unsqueezed_counts = unsqueeze(&unsqueezed_count_1, dim_neg_3);

        // Calculate the update term: delta / doubly_unsqueezed_counts
        // This performs element-wise division with broadcasting.
        let update_term = delta_i / &doubly_unsqueezed_counts;

        // Calculate the updated memory matrix: mm + update_term
        let updated_mm_i = mm_i + &update_term;
        updated_layers_mm.push(updated_mm_i);
    }

    // 3. Update iteration count and growth threshold
    let new_t = t + 1;
    // Ensure T::from works for usize -> T conversion
    let new_t_float = T::from(new_t).expect("Failed to convert usize to T");
    let new_growth_threshold = a / (new_t_float + a);

    (updated_layers_mm, new_t, new_growth_threshold)
}
