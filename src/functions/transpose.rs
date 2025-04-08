use ndarray::{ArrayD, Axis, IxDyn};

/// Transposes an ndarray ArrayD by swapping the specified dimensions.
///
/// # Arguments
///
/// * `arr` - The input array to transpose
/// * `dim1` - First dimension to swap
/// * `dim2` - Second dimension to swap
///
/// # Returns
///
/// A new transposed ArrayD
pub fn transpose_dims(arr: &ArrayD<f32>, dim1: usize, dim2: usize) -> ArrayD<f32> {
    let mut axes: Vec<usize> = (0..arr.ndim()).collect();

    // Swap the specified dimensions
    if dim1 < axes.len() && dim2 < axes.len() {
        axes.swap(dim1, dim2);
    } else {
        panic!(
            "Dimensions out of bounds: dim1={}, dim2={}, ndim={}",
            dim1,
            dim2,
            arr.ndim()
        );
    }

    arr.clone().permuted_axes(axes)
}
