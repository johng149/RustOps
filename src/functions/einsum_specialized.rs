use std::ops::Mul;

use ndarray::{Array, ArrayD, ArrayViewD, Axis, Dimension, IxDyn};
use ndarray::{LinalgScalar, ShapeError}; // Use IxDyn for dynamic dimensions // Use ShapeError for Result

// Helper trait bound for conciseness
pub trait EinsumScalar: LinalgScalar + Mul<Output = Self> + Copy {}
impl<T: LinalgScalar + Mul<Output = T> + Copy> EinsumScalar for T {}

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

//-------------------------------------------------------------------------
// einsum: bpcd,bpcdk->bpck
// Sum over 'd'
//-------------------------------------------------------------------------
pub fn einsum_bpcd_bpcdk_bpck_dyn<T>(
    a: &ArrayViewD<T>, // bpcd
    b: &ArrayViewD<T>, // bpcdk
) -> Result<ArrayD<T>, ShapeError>
where
    T: EinsumScalar,
{
    // 1. Get shapes and check dimensions
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a.ndim() != 4 || b.ndim() != 5 {
        panic!(
            "Input arrays must have dimensions 4 and 5 respectively. Got {} and {}",
            a.ndim(),
            b.ndim()
        );
    }

    // Check matching dimensions b, p, c, d
    // a: (b, p, c, d)       -> indices 0, 1, 2, 3
    // b: (b, p, c, d, k)    -> indices 0, 1, 2, 3, 4
    if a_shape[0] != b_shape[0] // b
        || a_shape[1] != b_shape[1] // p
        || a_shape[2] != b_shape[2] // c
        || a_shape[3] != b_shape[3]
    // d
    {
        panic!(
            "Dimension mismatch: A({:?}), B({:?}). b (0), p (1), c (2), and d (3) dimensions must match.",
            a_shape, b_shape
        );
    }

    let b_dim = a_shape[0];
    let p_dim = a_shape[1];
    let c_dim = a_shape[2];
    let d_dim = a_shape[3]; // Summation dimension
    let k_dim = b_shape[4]; // Dimension unique to b and output

    // 2. Reshape A to enable broadcasting with B's 'k' dimension
    // a: (b, p, c, d) -> a_reshaped: (b, p, c, d, 1)
    let a_reshaped_view = a.view().insert_axis(Axis(4));

    // Ensure the reshaped dimensions are as expected (optional sanity check)
    let expected_a_shape: &[usize] = &[b_dim, p_dim, c_dim, d_dim, 1];
    assert_eq!(
        a_reshaped_view.shape(),
        expected_a_shape,
        "Unexpected shape after reshaping A"
    );

    // 3. Element-wise multiplication with broadcasting
    // A_reshaped (b,p,c,d,1) * B (b,p,c,d,k) -> Intermediate (b,p,c,d,k)
    // Broadcasting happens along the last axis (k).
    let intermediate = &a_reshaped_view * b; // Note: b is already a view

    // 4. Sum over the 'd' dimension (axis 3)
    // Intermediate (b,p,c,d,k) -> Result (b,p,c,k)
    let result = intermediate.sum_axis(Axis(3)); // Axis(3) corresponds to 'd'

    // Ensure the final shape is correct (optional sanity check)
    let expected_result_shape: &[usize] = &[b_dim, p_dim, c_dim, k_dim];
    assert_eq!(
        result.shape(),
        expected_result_shape,
        "Unexpected final shape"
    );

    // 5. Convert result to ArrayD before returning
    Ok(result.into_dimensionality::<IxDyn>()?)
}

//-------------------------------------------------------------------------
// einsum: bhcjk,bhck->bhj
// Sum over 'c' and 'k'
//-------------------------------------------------------------------------
pub fn einsum_bhcjk_bhck_bhj_dyn<T>(
    a: &ArrayViewD<T>, // bhcjk
    b: &ArrayViewD<T>, // bhck
) -> Result<ArrayD<T>, ShapeError>
where
    T: EinsumScalar,
{
    // 1. Get shapes and check dimensions
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a.ndim() != 5 || b.ndim() != 4 {
        panic!(
            "Input arrays must have dimensions 5 and 4 respectively. Got {} and {}",
            a.ndim(),
            b.ndim()
        );
    }

    // Check matching dimensions b, h, c, k
    // a: (b, h, c, j, k) -> indices 0, 1, 2, 3, 4
    // b: (b, h, c, k)    -> indices 0, 1, 2, 3
    if a_shape[0] != b_shape[0] // b
        || a_shape[1] != b_shape[1] // h
        || a_shape[2] != b_shape[2] // c (summation dim)
        || a_shape[4] != b_shape[3]
    // k (summation dim)
    {
        panic!(
            "Dimension mismatch: A({:?}), B({:?}). b (0), h (1), c (A[2], B[2]), and k (A[4], B[3]) dimensions must match.",
            a_shape, b_shape
        );
    }

    let b_dim = a_shape[0];
    let h_dim = a_shape[1];
    let c_dim = a_shape[2]; // Summation dimension
    let j_dim = a_shape[3]; // Dimension unique to a and output
    let k_dim = a_shape[4]; // Summation dimension == b_shape[3]

    // 2. Reshape B to enable broadcasting with A's 'j' dimension
    // b: (b, h, c, k) -> b_reshaped: (b, h, c, 1, k)
    // Insert axis for 'j' at index 3
    let b_reshaped_view = b.view().insert_axis(Axis(3));

    // Ensure the reshaped dimensions are as expected (optional sanity check)
    let expected_b_shape: &[usize] = &[b_dim, h_dim, c_dim, 1, k_dim];
    assert_eq!(
        b_reshaped_view.shape(),
        expected_b_shape,
        "Unexpected shape after reshaping B"
    );

    // 3. Element-wise multiplication with broadcasting
    // A (b,h,c,j,k) * B_reshaped (b,h,c,1,k) -> Intermediate (b,h,c,j,k)
    // Broadcasting happens along the 'j' axis (index 3).
    let intermediate = a * &b_reshaped_view;

    // 4. Sum over the 'k' dimension (axis 4) FIRST
    // Intermediate (b,h,c,j,k) -> Intermediate2 (b,h,c,j)
    let intermediate2 = intermediate.sum_axis(Axis(4)); // Axis(4) corresponds to 'k'

    // 5. Sum over the 'c' dimension (axis 2) SECOND
    // Intermediate2 (b,h,c,j) -> Result (b,h,j)
    let result = intermediate2.sum_axis(Axis(2)); // Axis(2) corresponds to 'c'

    // Ensure the final shape is correct (optional sanity check)
    let expected_result_shape: &[usize] = &[b_dim, h_dim, j_dim];
    assert_eq!(
        result.shape(),
        expected_result_shape,
        "Unexpected final shape"
    );

    // 6. Convert result to ArrayD before returning
    Ok(result.into_dimensionality::<IxDyn>()?)
}

//-------------------------------------------------------------------------
// einsum: bncd,bnm->ncmd
// Sum over 'b'
// Requires permutation implicitly. Achieved via broadcasting alignment.
//-------------------------------------------------------------------------
pub fn einsum_bncd_bnm_ncmd_dyn<T>(
    a: &ArrayViewD<T>, // bncd
    b: &ArrayViewD<T>, // bnm
) -> Result<ArrayD<T>, ShapeError>
where
    T: EinsumScalar,
{
    // 1. Get shapes and check dimensions
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a.ndim() != 4 || b.ndim() != 3 {
        panic!(
            "Input arrays must have dimensions 4 and 3 respectively. Got {} and {}",
            a.ndim(),
            b.ndim()
        );
    }

    // Check matching dimensions b, n
    // a: (b, n, c, d) -> indices 0, 1, 2, 3
    // b: (b, n, m)    -> indices 0, 1, 2
    if a_shape[0] != b_shape[0] // b (summation dim)
        || a_shape[1] != b_shape[1]
    // n
    {
        panic!(
            "Dimension mismatch: A({:?}), B({:?}). b (0) and n (1) dimensions must match.",
            a_shape, b_shape
        );
    }

    let b_dim = a_shape[0]; // Summation dimension
    let n_dim = a_shape[1];
    let c_dim = a_shape[2];
    let d_dim = a_shape[3];
    let m_dim = b_shape[2];

    // 2. Reshape A and B to align dimensions for broadcasting and summation
    // Goal: Multiply element-wise to get shape (b, n, c, m, d), then sum over 'b' (axis 0)
    // a: (b, n, c, d) -> a_reshaped: (b, n, c, 1, d) (add axis for m)
    // b: (b, n, m)    -> b_reshaped: (b, n, 1, m, 1) (add axes for c, d)

    let a_reshaped_view = a.view().insert_axis(Axis(3)); // Insert axis for 'm'
    let b_reshaped_view = b.view().insert_axis(Axis(2)).insert_axis(Axis(4)); // Insert axes for 'c' and 'd'

    // Ensure the reshaped dimensions are as expected (optional sanity check)
    let expected_a_shape: &[usize] = &[b_dim, n_dim, c_dim, 1, d_dim];
    let expected_b_shape: &[usize] = &[b_dim, n_dim, 1, m_dim, 1];
    assert_eq!(
        a_reshaped_view.shape(),
        expected_a_shape,
        "Unexpected shape after reshaping A"
    );
    assert_eq!(
        b_reshaped_view.shape(),
        expected_b_shape,
        "Unexpected shape after reshaping B"
    );

    // 3. Element-wise multiplication with broadcasting
    // A_reshaped (b,n,c,1,d) * B_reshaped (b,n,1,m,1) -> Intermediate (b,n,c,m,d)
    let intermediate = &a_reshaped_view * &b_reshaped_view;

    // 4. Sum over the 'b' dimension (axis 0)
    // Intermediate (b,n,c,m,d) -> Result (n,c,m,d)
    let result = intermediate.sum_axis(Axis(0)); // Axis(0) corresponds to 'b'

    // Ensure the final shape is correct (optional sanity check)
    let expected_result_shape: &[usize] = &[n_dim, c_dim, m_dim, d_dim];
    assert_eq!(
        result.shape(),
        expected_result_shape,
        "Unexpected final shape"
    );

    // 5. Convert result to ArrayD before returning
    Ok(result.into_dimensionality::<IxDyn>()?)
}

//-------------------------------------------------------------------------
// einsum: ncmd,bnm->bncd
// Sum over 'm'
// Requires permutation implicitly. Achieved via broadcasting alignment.
//-------------------------------------------------------------------------
pub fn einsum_ncmd_bnm_bncd_dyn<T>(
    a: &ArrayViewD<T>, // ncmd
    b: &ArrayViewD<T>, // bnm
) -> Result<ArrayD<T>, ShapeError>
where
    T: EinsumScalar,
{
    // 1. Get shapes and check dimensions
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a.ndim() != 4 || b.ndim() != 3 {
        panic!(
            "Input arrays must have dimensions 4 and 3 respectively. Got {} and {}",
            a.ndim(),
            b.ndim()
        );
    }

    // Check matching dimensions n, m
    // a: (n, c, m, d) -> indices 0, 1, 2, 3
    // b: (b, n, m)    -> indices 0, 1, 2
    if a_shape[0] != b_shape[1] // n
        || a_shape[2] != b_shape[2]
    // m (summation dim)
    {
        panic!(
            "Dimension mismatch: A({:?}), B({:?}). n (A[0], B[1]) and m (A[2], B[2]) dimensions must match.",
            a_shape, b_shape
        );
    }

    let n_dim = a_shape[0]; // == b_shape[1]
    let c_dim = a_shape[1];
    let m_dim = a_shape[2]; // Summation dimension == b_shape[2]
    let d_dim = a_shape[3];
    let b_dim = b_shape[0]; // Dimension unique to b and output

    // 2. Reshape A and B to align dimensions for broadcasting and summation
    // Goal: Multiply element-wise to get shape (b, n, c, m, d), then sum over 'm' (axis 3)
    // a: (n, c, m, d) -> a_reshaped: (1, n, c, m, d) (add axis for b)
    // b: (b, n, m)    -> b_reshaped: (b, n, 1, m, 1) (add axes for c, d)

    let a_reshaped_view = a.view().insert_axis(Axis(0)); // Insert axis for 'b' at the beginning
    let b_reshaped_view = b.view().insert_axis(Axis(2)).insert_axis(Axis(4)); // Insert axes for 'c' and 'd'

    // Ensure the reshaped dimensions are as expected (optional sanity check)
    let expected_a_shape: &[usize] = &[1, n_dim, c_dim, m_dim, d_dim];
    let expected_b_shape: &[usize] = &[b_dim, n_dim, 1, m_dim, 1];
    assert_eq!(
        a_reshaped_view.shape(),
        expected_a_shape,
        "Unexpected shape after reshaping A"
    );
    assert_eq!(
        b_reshaped_view.shape(),
        expected_b_shape,
        "Unexpected shape after reshaping B"
    );

    // 3. Element-wise multiplication with broadcasting
    // A_reshaped (1,n,c,m,d) * B_reshaped (b,n,1,m,1) -> Intermediate (b,n,c,m,d)
    let intermediate = &a_reshaped_view * &b_reshaped_view;

    // 4. Sum over the 'm' dimension (axis 3)
    // Intermediate (b,n,c,m,d) -> Result (b,n,c,d)
    let result = intermediate.sum_axis(Axis(3)); // Axis(3) corresponds to 'm'

    // Ensure the final shape is correct (optional sanity check)
    let expected_result_shape: &[usize] = &[b_dim, n_dim, c_dim, d_dim];
    assert_eq!(
        result.shape(),
        expected_result_shape,
        "Unexpected final shape"
    );

    // 5. Convert result to ArrayD before returning
    Ok(result.into_dimensionality::<IxDyn>()?)
}
