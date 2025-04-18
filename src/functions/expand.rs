use ndarray::{Array, ArrayD, IxDyn, NdFloat};
use std::convert::TryFrom;

use ndarray::prelude::*;

/// Expands a tensor along a specified dimension by repeating its values.
///
/// This function implements functionality similar to PyTorch's unsqueeze and expand operations.
/// It inserts a new axis at the specified dimension and then repeats the tensor's values
/// along that new dimension.
///
/// # Arguments
///
/// * `input` - The input ArrayD to expand
/// * `dim` - The dimension at which to insert and expand
/// * `size` - The size to expand to along the new dimension
///
/// # Returns
///
/// A new ArrayD with an expanded dimension
pub fn expand_at_dim<A>(
    input: &ArrayD<A>,
    dim: usize,
    size: usize,
) -> Result<ArrayD<A>, &'static str>
where
    A: Clone,
{
    // Check that dimension is valid
    if dim >= input.ndim() {
        return Err("Dimension index out of bounds");
    }

    // Build shape for the expansion
    let mut expanded_shape = input.shape().to_vec();
    expanded_shape[dim] = size;

    // Use broadcast to perform the expansion
    let view = match input.broadcast(IxDyn(&expanded_shape)) {
        Some(view) => view,
        None => return Err("Failed to broadcast tensor for expand operation"),
    };

    // Create an owned array from the view
    Ok(view.to_owned())
}
