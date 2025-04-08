use ndarray::{Array, ArrayBase, Data, Dimension, IxDyn, ShapeBuilder};
use std::fmt::Debug;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/src/functions/reshape.rs

/// Error types for the reshape function.
#[derive(Debug, PartialEq)]
pub enum ReshapeError {
    /// The new shape is incompatible with the input array's size.
    IncompatibleShape,
    /// The -1 (inferred dimension) appears more than once in the shape.
    MultipleInferredDimensions,
}

/// Reshapes an array to a new shape.
/// Mimics the behavior of PyTorch's `torch.reshape` or NumPy's `np.reshape`.
///
/// # Arguments
///
/// * `input`: The input array.
/// * `shape`: The new shape as a slice of i64. Negative values (except -1) will be treated as
///   counting from the end of the shape. A value of -1 will be inferred from the size of the array.
///
/// # Returns
///
/// * `Ok(Array<A, IxDyn>)`: The reshaped array.
/// * `Err(ReshapeError)`: If the input shape is incompatible or invalid.
///
/// # Type Parameters
///
/// * `A`: The element type of the array.
/// * `S`: The data storage type.
/// * `D`: The dimension type of the input array.
pub fn reshape<A, S, D>(
    input: &ArrayBase<S, D>,
    shape: &[i64],
) -> Result<Array<A, IxDyn>, ReshapeError>
where
    A: Clone,
    S: Data<Elem = A>,
    D: Dimension,
{
    let original_size = input.len();
    let mut new_shape = Vec::with_capacity(shape.len());
    let mut inferred_index = None;
    let mut known_size = 1;

    // First pass: process all dimensions except the inferred one (-1)
    for (idx, &dim) in shape.iter().enumerate() {
        if dim == -1 {
            if inferred_index.is_some() {
                return Err(ReshapeError::MultipleInferredDimensions);
            }
            inferred_index = Some(idx);
            new_shape.push(0); // Placeholder for now
        } else if dim < 0 {
            // Handle negative indices (counting from the end)
            // Not applicable for reshape, but included for API compatibility
            return Err(ReshapeError::IncompatibleShape);
        } else {
            new_shape.push(dim as usize);
            known_size *= dim as usize;
        }
    }

    // Second pass: infer the dimension if needed
    if let Some(idx) = inferred_index {
        if known_size == 0 {
            return Err(ReshapeError::IncompatibleShape);
        }

        // Infer the dimension
        if original_size % known_size != 0 {
            return Err(ReshapeError::IncompatibleShape);
        }

        new_shape[idx] = original_size / known_size;
    } else if known_size != original_size {
        // If no dimension needs to be inferred, check that the total size matches
        return Err(ReshapeError::IncompatibleShape);
    }

    // Perform the reshape
    let reshaped = input.to_owned().into_shape(IxDyn(&new_shape))?;
    Ok(Array::from_shape_vec(
        IxDyn(&new_shape),
        reshaped.iter().cloned().collect(),
    )?)
}

// Provide a conversion from ndarray's ShapeError to our ReshapeError
impl From<ndarray::ShapeError> for ReshapeError {
    fn from(_: ndarray::ShapeError) -> Self {
        ReshapeError::IncompatibleShape
    }
}
