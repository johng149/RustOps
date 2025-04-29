use ndarray::ShapeError;
use ndarray::{Array, ArrayD, ArrayViewD, Axis, Dimension, IxDyn}; // Use IxDyn for dynamic dimensions // Use ShapeError for Result

// Generic implementation using ArrayViewD
pub fn einsum_bfmd_bfd_bfm_dyn<T>(
    a: &ArrayViewD<T>,
    b: &ArrayViewD<T>,
) -> Result<ArrayD<T>, ShapeError>
where
    T: ndarray::LinalgScalar + std::ops::Mul<Output = T> + Copy, // Added Copy constraint often needed with LinalgScalar operations
{
    // 1. Get shapes and check dimensions
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a.ndim() != 4 || b.ndim() != 3 {
        // Consider returning a specific error instead of panicking
        panic!(
            "Input arrays must have dimensions 4 and 3 respectively. Got {} and {}",
            a.ndim(),
            b.ndim()
        );
    }

    // Check matching dimensions b, f, d
    // a: (b, f, m, d) -> indices 0, 1, 2, 3
    // b: (b, f, d)    -> indices 0, 1, 2
    if a_shape[0] != b_shape[0] || a_shape[1] != b_shape[1] || a_shape[3] != b_shape[2] {
        // Consider returning a specific error
        panic!(
            "Dimension mismatch: A({:?}), B({:?}). b (0), f (1), and d (A[3], B[2]) dimensions must match.",
            a_shape, b_shape
        );
    }

    let b_dim = a_shape[0];
    let f_dim = a_shape[1];
    let m_dim = a_shape[2];
    let d_dim = a_shape[3]; // == b_shape[2]

    // 2. Reshape B to enable broadcasting: (B, F, D) -> (B, F, 1, D)
    // Insert a new axis at index 2
    let b_reshaped_view = b.view().insert_axis(Axis(2)); // Shape becomes (B, F, 1, D)

    // Ensure the reshaped dimensions are as expected (optional sanity check)
    let expected_b_shape: &[usize] = &[b_dim, f_dim, 1, d_dim];
    assert_eq!(
        b_reshaped_view.shape(),
        expected_b_shape,
        "Unexpected shape after reshaping B"
    );

    // 3. Element-wise multiplication with broadcasting
    // A (B, F, M, D) * B_reshaped (B, F, 1, D) -> Intermediate (B, F, M, D)
    // Broadcasting happens automatically.
    let intermediate = a * &b_reshaped_view;

    // 4. Sum over the 'd' dimension (the last axis, index 3)
    // Intermediate (B, F, M, D) -> Result (B, F, M)
    let result = intermediate.sum_axis(Axis(3)); // Axis(3) corresponds to 'd'

    // Ensure the final shape is correct (optional sanity check)
    let expected_result_shape: &[usize] = &[b_dim, f_dim, m_dim];
    assert_eq!(
        result.shape(),
        expected_result_shape,
        "Unexpected final shape"
    );

    // Convert result to ArrayD before returning
    Ok(result.into_dimensionality::<IxDyn>()?) // Use ? to propagate potential ShapeError
}
