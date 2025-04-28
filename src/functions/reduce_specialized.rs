use ndarray::{
    Array, Array2, Array3, ArrayD, ArrayViewD, Axis, Dimension, Ix2, Ix3, LinalgScalar, ShapeError,
};
use std::error::Error;
use std::fmt;

// Define a custom error type for dimension mismatches
#[derive(Debug)]
pub enum ReduceError {
    DimensionMismatch { expected: usize, actual: usize },
    ShapeError(ShapeError), // To wrap ndarray's errors if needed
}

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReduceError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Input array has incorrect dimensions: expected {}, got {}",
                    expected, actual
                )
            }
            ReduceError::ShapeError(e) => write!(f, "ndarray shape error: {}", e),
        }
    }
}

impl Error for ReduceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReduceError::ShapeError(e) => Some(e),
            _ => None,
        }
    }
}

// Helper macro for dimension checks
macro_rules! check_ndim {
    ($arr:expr, $expected_ndim:expr) => {
        let actual_ndim = $arr.ndim();
        if actual_ndim != $expected_ndim {
            return Err(ReduceError::DimensionMismatch {
                expected: $expected_ndim,
                actual: actual_ndim,
            });
        }
    };
}

/// Reduces 'bnm -> nm' by summing over the first axis (b).
/// Expects a 3D input array. Returns a dynamically dimensioned array (ArrayD).
pub fn reduce_bnm_to_nm<T>(input: &ArrayViewD<T>) -> Result<ArrayD<T>, ReduceError>
where
    T: LinalgScalar,
{
    check_ndim!(input, 3);
    // Sum over axis 0 (b)
    Ok(input.sum_axis(Axis(0)).into_dyn())
}

/// Reduces 'bhck -> bh' by summing over the last two axes (c and k).
/// Expects a 4D input array. Returns a dynamically dimensioned array (ArrayD).
pub fn reduce_bhck_to_bh<T>(input: &ArrayViewD<T>) -> Result<ArrayD<T>, ReduceError>
where
    T: LinalgScalar,
{
    check_ndim!(input, 4);
    // Sum over axis 3 (k), then axis 2 (c)
    // Summing highest index first avoids shifting indices of subsequent sums
    let summed_k = input.sum_axis(Axis(3)); // Shape: bhc
    let summed_c = summed_k.sum_axis(Axis(2)); // Shape: bh
    Ok(summed_c.into_dyn())
}

/// Reduces 'bfd -> bf' by summing over the last axis (d).
/// Expects a 3D input array. Returns a dynamically dimensioned array (ArrayD).
pub fn reduce_bfd_to_bf<T>(input: &ArrayViewD<T>) -> Result<ArrayD<T>, ReduceError>
where
    T: LinalgScalar,
{
    check_ndim!(input, 3);
    // Sum over axis 2 (d)
    Ok(input.sum_axis(Axis(2)).into_dyn())
}

/// Reduces 'bfmd -> bfm' by summing over the last axis (d).
/// Expects a 4D input array. Returns a dynamically dimensioned array (ArrayD).
pub fn reduce_bfmd_to_bfm<T>(input: &ArrayViewD<T>) -> Result<ArrayD<T>, ReduceError>
where
    T: LinalgScalar,
{
    check_ndim!(input, 4);
    // Sum over axis 3 (d)
    Ok(input.sum_axis(Axis(3)).into_dyn())
}
