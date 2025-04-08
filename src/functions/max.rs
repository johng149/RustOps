use ndarray::{Array, ArrayBase, ArrayView, Axis, Data, Dimension, Ix1, IxDyn, RemoveAxis};
use std::cmp::Ordering;
use std::fmt::Debug;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/src/functions/max.rs

/// Error types for the max function.
#[derive(Debug, PartialEq)]
pub enum MaxError {
    /// The input array is empty (when finding the overall max).
    EmptyInput,
    /// The specified dimension has size 0.
    ZeroDimSize(usize), // Contains the axis index
    /// The specified axis index is out of bounds.
    InvalidAxis(usize), // Contains the axis index
}

/// Finds the maximum values and their indices along a given dimension.
/// Mimics the behavior of PyTorch's `torch.max` when used with a `dim` parameter.
///
/// # Arguments
///
/// * `input`: The input array.
/// * `dim`: The dimension along which to find the maximum values and indices.
///   - Must be provided (unlike argmax, max requires a dimension).
///
/// # Returns
///
/// * `Ok((Array<A, D::Smaller>, Array<i64, D::Smaller>))`: A tuple containing:
///   - The array of maximum values along the specified dimension.
///   - The array of indices (as `i64`) where maximum values occur.
/// * `Err(MaxError)`: If the input is invalid (e.g., empty, zero-sized dimension).
///
/// # Type Parameters
///
/// * `A`: The element type of the array. Must implement `PartialOrd` for comparison and `Copy`
///        for efficient processing within closures.
/// * `S`: The data storage type (e.g., `OwnedRepr<A>`, `ViewRepr<&'a A>`).
/// * `D`: The dimension type of the input array.
///
/// # Panics
///
/// This function generally avoids panics and returns `Result`. However, internal `ndarray`
/// operations or extreme resource exhaustion could potentially cause panics. Also, the cast
/// from `usize` to `i64` could theoretically panic on 32-bit systems if the index exceeds
/// `i64::MAX`, although this is highly unlikely for typical array dimensions.
///
/// # NaN Handling Note
///
/// This implementation uses `partial_cmp`. If the input contains NaN values, the behavior might
/// differ slightly from PyTorch's `max`, which has specific NaN propagation/handling rules.
pub fn max<A, S, D>(
    input: &ArrayBase<S, D>,
    dim: usize,
) -> Result<(Array<A, D::Smaller>, Array<i64, D::Smaller>), MaxError>
where
    A: PartialOrd + Copy,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    let ndim = input.ndim();
    if dim >= ndim {
        return Err(MaxError::InvalidAxis(dim));
    }

    let axis = Axis(dim);
    let dim_size = input.shape()[dim];

    if dim_size == 0 {
        return Err(MaxError::ZeroDimSize(dim));
    }

    if input.len() == 0 {
        return Err(MaxError::EmptyInput);
    }

    // Create arrays to hold the max values and indices
    let max_values = input.map_axis(axis, |view: ArrayView<A, Ix1>| {
        view.fold(
            view[0],
            |max_val, &val| {
                if val > max_val { val } else { max_val }
            },
        )
    });

    let max_indices = input.map_axis(axis, |view: ArrayView<A, Ix1>| {
        let (idx, _) = view
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less))
            .unwrap(); // Safe due to dim_size > 0 check
        idx as i64
    });

    Ok((max_values, max_indices))
}
