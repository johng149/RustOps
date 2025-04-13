use ndarray::{ArrayD, IxDyn};

/// Unsqueeze a tensor by adding a dimension of size 1 at the specified index.
///
/// # Arguments
///
/// * `input` - The input tensor.
/// * `dim` - The index where to add the new dimension.
///
/// # Returns
///
/// The unsqueezed tensor.
pub fn unsqueeze(input: &ArrayD<f32>, dim: usize) -> ArrayD<f32> {
    let input_shape = input.shape();

    // Create a new shape with an additional dimension of size 1 at the specified index
    let mut new_shape: Vec<usize> = input_shape.to_vec();

    // Check if the dimension index is valid (can be at most equal to the length of the shape)
    if dim <= new_shape.len() {
        new_shape.insert(dim, 1);
    } else {
        // If the dimension is out of bounds, return the tensor unchanged
        return input.clone();
    }

    // Reshape the tensor
    input.clone().into_shape(IxDyn(&new_shape)).unwrap()
}
