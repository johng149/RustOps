use super::expand::expand_at_dim;
use super::reshape::reshape;
use super::slicing::slice_second_dim;
use super::transpose::transpose_dims;
use super::unsqueeze::unsqueeze;
use ndarray::ArrayD;
use std::error::Error;
use std::fmt::Debug;

/// Expands a 2D array to support a specific batch size by replicating columns as needed
///
/// # Arguments
/// * `x` - A 2D input array with shape [nodes, mems]
/// * `batch_size` - The desired batch size for the second dimension
///
/// # Returns
/// A new array with shape [nodes, batch_size]
pub fn expand_for_batches<T>(x: &ArrayD<T>, batch_size: usize) -> Result<ArrayD<T>, Box<dyn Error>>
where
    T: Clone + Debug,
{
    // Ensure x is 2D
    if x.ndim() != 2 {
        return Err("Input array must be 2D".into());
    }

    let shape = x.shape();
    let nodes = shape[0];
    let mems = shape[1];

    // Calculate full_expands
    let full_expands = (batch_size / mems) + 1;

    // 1. Unsqueeze: add a dimension at the front
    let unsqueezed = unsqueeze(x, 0);

    // 2. Expand: replicate along the first dimension
    let expanded = expand_at_dim(&unsqueezed, 0, full_expands)?;

    // 3. Transpose: swap the first and second dimensions
    let transposed = transpose_dims(&expanded, 0, 1);

    // 4. Reshape: reshape to (nodes, -1)
    let shape = &[nodes, full_expands * mems];
    let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    let reshaped = reshape(&transposed, &shape_i64).unwrap();

    // 5. Slice: take only the first batch_size elements from the second dimension
    let result = slice_second_dim(&reshaped, batch_size).unwrap();

    Ok(result)
}
