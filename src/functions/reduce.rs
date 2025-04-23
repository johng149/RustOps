use crate::functions::einsum::einsum_ndarray_dyn;
use crate::functions::ones::ones;
use ndarray::{Array, ArrayD, Axis, Dimension, IntoDimension, IxDyn, NdFloat};
use num_traits::One;

/// Performs a reduction operation similar to einsum but with an array of ones.
///
/// # Arguments
///
/// * `input` - The input array to reduce.
/// * `equation` - The einsum-like equation representing the reduction pattern.
///
/// # Returns
///
/// A reduced array according to the specified equation.
pub fn reduce<T>(input: &ArrayD<T>, equation: &str) -> Result<ArrayD<T>, &'static str>
where
    T: NdFloat + One + Copy,
{
    // Create an array of ones with the same shape as the input
    let ones_array = Array::<T, _>::ones(input.shape());

    let tensors = [input, &ones_array];
    einsum_ndarray_dyn(equation, &tensors)
}

/// Helper function to create an ArrayD from a vector and shape
pub fn array_from_shape_vec(shape: &[usize], data: Vec<f32>) -> ArrayD<f32> {
    Array::from_shape_vec(IxDyn(shape), data)
        .expect("Failed to create array with specified shape and data")
}
