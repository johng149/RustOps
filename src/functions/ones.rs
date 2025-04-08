use ndarray::{IntoDimension, prelude::*};

/// Creates an array filled with ones of the specified shape.
///
/// # Arguments
///
/// * `shape` - Shape of the array to create.
///
/// # Returns
///
/// An ndarray Array of the specified shape filled with ones.
pub fn ones<D>(shape: D) -> Array<f32, D::Dim>
where
    D: IntoDimension,
{
    Array::ones(shape)
}
