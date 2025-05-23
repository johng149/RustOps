use ndarray::{Array, ArrayD, Axis, Dimension, IxDyn, ShapeError, s};
use num_traits::{AsPrimitive, Num, Zero}; // Added AsPrimitive for pad_value if needed

#[derive(Debug)]
pub enum ChunkingError {
    InvalidInputShape(String),
    InvalidChunkSize(String),
    NdarrayShapeError(ShapeError),
    // Add other error types as needed
}

impl From<ShapeError> for ChunkingError {
    fn from(err: ShapeError) -> ChunkingError {
        ChunkingError::NdarrayShapeError(err)
    }
}

/// Reshapes a 4D tensor (batch, height, width, channels) into chunks.
///
/// Args:
///     tensor: Input tensor of shape (batch, height, width, channels).
///     chunk_height: The height of each chunk.
///     chunk_width: The width of each chunk.
///     pad_value: Value used for padding if the dimensions are not perfectly divisible.
///
/// Returns:
///     A 3D tensor of shape (batch, num_total_chunks, chunk_height * chunk_width).
///     Chunks are ordered by width_step, then height_step, then channel.
pub fn chunking<T>(
    tensor: &ArrayD<T>,
    chunk_height: usize,
    chunk_width: usize,
    pad_value: T,
) -> Result<ArrayD<T>, ChunkingError>
where
    T: Num + Copy + Zero, // Zero trait for pad_value, Copy for element copying
{
    // 1. Get shape and validate
    if tensor.ndim() != 4 {
        return Err(ChunkingError::InvalidInputShape(
            "Input tensor must be 4D (batch, height, width, channels)".to_string(),
        ));
    }
    let shape = tensor.shape(); // shape() returns &[usize]
    let batch = shape[0];
    let height = shape[1];
    let width = shape[2];
    let channels = shape[3];

    if chunk_height == 0 || chunk_width == 0 {
        return Err(ChunkingError::InvalidChunkSize(
            "chunk_height and chunk_width must be positive.".to_string(),
        ));
    }

    // 2. Calculate padding dimensions
    let num_h_steps = (height + chunk_height - 1) / chunk_height;
    let height_padded = num_h_steps * chunk_height;
    let num_w_steps = (width + chunk_width - 1) / chunk_width;
    let width_padded = num_w_steps * chunk_width;

    // let _height_needed_pad = height_padded - height; // For reference
    // let _width_needed_pad = width_padded - width;   // For reference

    // 3. Pad the tensor
    let padded_dim_slice = [batch, height_padded, width_padded, channels];
    let mut padded_tensor = Array::from_elem(IxDyn(&padded_dim_slice), pad_value);

    // Create a view of the original tensor with explicit 4D shape for assignment
    let original_tensor_view_4d = tensor.view().into_dimensionality::<ndarray::Ix4>()?; // Convert &ArrayD<T> to ArrayView<T, Ix4>

    // Copy original data into the padded tensor
    padded_tensor
        .slice_mut(s![.., 0..height, 0..width, ..])
        .assign(&original_tensor_view_4d);

    // 4. Reshape to (batch, num_h_steps, chunk_height, num_w_steps, chunk_width, channels)
    let reshaped_dims_slice = [
        batch,
        num_h_steps,
        chunk_height,
        num_w_steps,
        chunk_width,
        channels,
    ];
    let reshaped = padded_tensor.into_shape(IxDyn(&reshaped_dims_slice))?;

    // 5. Permute
    // Python: permuted = reshaped.permute(0, 3, 1, 5, 2, 4)
    // Original indices: (0:B, 1:N_h, 2:C_h, 3:N_w, 4:C_w, 5:Chan)
    // New order:        (0:B, 3:N_w, 1:N_h, 5:Chan, 2:C_h, 4:C_w)
    let permute_order: &[usize] = &[0, 3, 1, 5, 2, 4];
    // permute_axes returns an ArrayView. We need an owned array for the next reshape.
    let permuted = reshaped.permuted_axes(permute_order).to_owned();

    // 6. Reshape to (batch, num_total_chunks, chunk_area)
    let num_total_chunks = num_w_steps * num_h_steps * channels;
    let chunk_area = chunk_height * chunk_width;

    let output_dims_slice = [batch, num_total_chunks, chunk_area];
    let output_tensor = permuted
        .as_standard_layout()
        .to_owned()
        .into_shape(IxDyn(&output_dims_slice))?;

    Ok(output_tensor) // output_tensor is Array<T, IxDyn>, which is type alias for ArrayD<T>
}
