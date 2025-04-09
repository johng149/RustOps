use crate::functions::reshape::{ReshapeError, reshape};
use crate::functions::transpose::transpose_dims;
use ndarray::Array;
use ndarray::{ArrayD, IxDyn};

// filepath: /media/john/Tertiary/Projects/ML/RustOps/src/functions/rearrange.rs

/// Rearrange a 3D tensor with pattern 'batch mems flag -> mems (batch flag)'
/// This is equivalent to einops' rearrange operation that reorders and reshapes dimensions.
///
/// # Arguments
///
/// * `input` - The input array with shape [batch, mems, flag]
///
/// # Returns
///
/// * `Ok(ArrayD<f32>)` - The rearranged array with shape [mems, batch*flag]
/// * `Err(ReshapeError)` - If the reshape operation fails
pub fn rearrange_batch_mems_flag(input: &ArrayD<f32>) -> Result<ArrayD<f32>, ReshapeError> {
    // Check that input has 3 dimensions
    if input.ndim() != 3 {
        return Err(ReshapeError::IncompatibleShape);
    }

    let shape = input.shape();
    let batch = shape[0];
    let mems = shape[1];
    let flag = shape[2];

    // First, transpose to get 'mems batch flag'
    let transposed = transpose_dims(input, 0, 1);

    // Then, reshape to combine batch and flag dimensions: 'mems (batch flag)'
    let new_shape = vec![mems as i64, (batch * flag) as i64];

    // Use our reshape function to get the final result
    let reshaped: ArrayD<f32> = reshape(&transposed, &new_shape)?;
    Ok(reshaped)
}
