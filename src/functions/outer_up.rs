use super::outer_forward_parallel::outer_forward_parallel;
use ndarray::{ArrayD, NdFloat};
use num_traits::Float;

/// Wrapper around outer_forward_parallel that accepts a list of memory tensors
/// and uses only the first element
///
/// # Arguments
/// * `mems` - List of memory tensors, only the first one will be used
/// * `xs` - Input tensor
/// * `rho` - Regularization parameter
///
/// # Returns
/// * Result of outer_forward_parallel on the first memory tensor
pub fn outer_up<T>(mems: Vec<ArrayD<T>>, xs: ArrayD<T>, rho: f32) -> ArrayD<T>
where
    T: Clone + std::fmt::Debug + ndarray::ScalarOperand + Float + NdFloat,
{
    // Extract the first memory tensor from the list
    // If the list is empty, panic (or you could return a Result type instead)
    let first_mem = mems.into_iter().next().expect("mems list cannot be empty");

    // Call the original function with the first memory tensor
    outer_forward_parallel(first_mem, xs, rho)
}
