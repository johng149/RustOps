use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::Zero;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/src/functions/zeros_like.rs

/// Creates an array filled with zeros with the same shape as the input array.
///
/// # Arguments
///
/// * `input` - Input array whose shape will be used for the new array.
///
/// # Returns
///
/// An ndarray Array of the same shape as input, filled with zeros.
pub fn zeros_like<T, S, D>(input: &ArrayBase<S, D>) -> Array<T, D>
where
    T: Zero + Clone,
    S: Data,
    D: Dimension,
{
    Array::zeros(input.raw_dim())
}
