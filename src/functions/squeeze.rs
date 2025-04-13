use ndarray::{ArrayD, IxDyn};

/// Squeeze a tensor along the specified dimension.
///
/// If the dimension at the specified index has size 1, it will be removed from the tensor.
/// If the dimension doesn't have size 1, the tensor is returned unchanged.
///
/// # Arguments
///
/// * `input` - The input tensor.
/// * `dim` - The dimension to squeeze.
///
/// # Returns
///
/// The squeezed tensor.
pub fn squeeze(input: &ArrayD<f32>, dim: usize) -> ArrayD<f32> {
    let input_shape = input.shape();

    // Check if the dimension is valid and has size 1
    if dim >= input_shape.len() || input_shape[dim] != 1 {
        return input.clone();
    }

    // Create a new shape without the specified dimension
    let mut new_shape: Vec<usize> = input_shape.to_vec();
    new_shape.remove(dim);

    // Reshape the tensor
    input.clone().into_shape(IxDyn(&new_shape)).unwrap()
}
