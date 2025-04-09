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
    let ndim = arr.ndim();
    let mut axes: Vec<usize> = (0..ndim).collect();

    // Check bounds before swapping
    if dim1 >= ndim || dim2 >= ndim {
        // Consider returning a Result instead of panicking for library code
        panic!(
            "Dimensions out of bounds: dim1={}, dim2={}, ndim={}",
            dim1, dim2, ndim
        );
    }

    // Swap the specified dimensions in the axes list
    axes.swap(dim1, dim2);

    // 1. Create a view with permuted axes. This doesn't copy data yet,
    //    just changes how the existing data is interpreted (shape and strides).
    let permuted_view = arr.clone().permuted_axes(axes);

    // 2. Create a new owned array from the view. `.to_owned()` on a view
    //    creates a new array with its own data buffer, arranged in the
    //    standard (C-order) layout corresponding to the view's shape.
    permuted_view.as_standard_layout().to_owned()
}
