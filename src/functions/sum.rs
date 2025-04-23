use ndarray::prelude::*;

/// Computes the sum of elements along a specified axis for ndarray arrays.
///
/// # Arguments
///
/// * `matrix` - Input array
/// * `axis` - Axis along which to perform the sum
/// * `keepdim` - If true, the output will have the same number of dimensions as the input,
///   with the specified axis reduced to size 1. If false, the specified axis will be removed.
///
/// # Returns
///
/// A new array with the sum computed along the specified axis.
pub fn sum_ndarray<T, D>(matrix: &Array<T, D>, axis: Axis, keepdim: bool) -> Array<T, IxDyn>
where
    T: num_traits::Float + std::ops::AddAssign + Clone,
    D: ndarray::RemoveAxis + ndarray::Dimension,
{
    let sum = matrix.sum_axis(axis);

    if !keepdim {
        return sum.into_dyn();
    }

    // Create a new shape with a singleton dimension at the axis position
    let mut new_shape = Vec::new();
    let mut orig_dim_idx = 0;

    for i in 0..matrix.ndim() {
        if i == axis.index() {
            new_shape.push(1);
        } else {
            new_shape.push(sum.shape()[orig_dim_idx]);
            orig_dim_idx += 1;
        }
    }

    // Reshape the sum to include the singleton dimension
    sum.into_shape(IxDyn(&new_shape)).unwrap()
}

/// Computes the sum of elements along a specified axis for ndarray arrays with any numeric type.
///
/// # Arguments
///
/// * `matrix` - Input array of any numeric type (including integers)
/// * `axis` - Axis along which to perform the sum
/// * `keepdim` - If true, the output will have the same number of dimensions as the input,
///   with the specified axis reduced to size 1. If false, the specified axis will be removed.
///
/// # Returns
///
/// A new array with the sum computed along the specified axis.
pub fn sum_generic<T, D>(matrix: &Array<T, D>, axis: Axis, keepdim: bool) -> Array<T, IxDyn>
where
    T: num_traits::Zero + std::ops::AddAssign + Clone,
    D: ndarray::RemoveAxis + ndarray::Dimension,
{
    let sum = matrix.sum_axis(axis);

    if !keepdim {
        return sum.into_dyn();
    }

    // Create a new shape with a singleton dimension at the axis position
    let mut new_shape = Vec::new();
    let mut orig_dim_idx = 0;

    for i in 0..matrix.ndim() {
        if i == axis.index() {
            new_shape.push(1);
        } else {
            new_shape.push(sum.shape()[orig_dim_idx]);
            orig_dim_idx += 1;
        }
    }

    // Reshape the sum to include the singleton dimension
    sum.into_shape(IxDyn(&new_shape)).unwrap()
}
