use ndarray::{Array, ArrayBase, ArrayView, Axis, Data, Dimension, Ix1, IxDyn, RemoveAxis, arr0};
use std::cmp::Ordering;
use std::fmt::Debug; // For Debug bound in error

/// Error types for the argmax function.
#[derive(Debug, PartialEq)]
pub enum ArgmaxError {
    /// The input array is empty (when finding the overall max).
    EmptyInput,
    /// The specified dimension has size 0.
    ZeroDimSize(usize), // Contains the axis index
    /// The specified axis index is out of bounds.
    InvalidAxis(usize), // Contains the axis index
}

/// Finds the indices (as i64) of the maximum values of an array along a given dimension.
/// Mimics the behavior of PyTorch's `torch.argmax`.
///
/// # Arguments
///
/// * `input`: The input array.
/// * `dim`: The dimension along which to find the maximum indices.
///   - If `None`, the input array is flattened, and the index of the single maximum value is returned.
///   - If `Some(axis_index)`, finds the maximum index along the specified axis.
/// * `keepdim`: Whether the output tensor has `dim` retained or not.
///   - If `false` (default), the `dim` is removed (or squeezed).
///   - If `true`, the `dim` is retained with size 1.
///   - This argument is ignored if `dim` is `None`.
///
/// # Returns
///
/// * `Ok(Array<i64, IxDyn>)`: An array containing the indices (as `i64`) of the maximum values.
///   The shape depends on `dim` and `keepdim`. The dimension type is `IxDyn` for flexibility.
/// * `Err(ArgmaxError)`: If the input is invalid (e.g., empty, zero-sized dimension).
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
/// differ slightly from PyTorch's `argmax`, which has specific NaN propagation/handling rules
/// (often documented as returning the index of the first `NaN` if encountered, but observed
/// behavior can sometimes prioritize non-NaN max values). This implementation will typically
/// *not* select a NaN as the maximum unless forced by the `unwrap_or` fallback, and its
/// position relative to other maximal values might affect the outcome if NaNs are present.
/// For precise NaN handling matching PyTorch, a more complex comparison logic would be needed.
pub fn argmax<A, S, D>(
    input: &ArrayBase<S, D>,
    dim: Option<usize>,
    keepdim: bool,
) -> Result<Array<i64, IxDyn>, ArgmaxError>
// <-- Changed return type to i64
where
    A: PartialOrd + Copy,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    // Handle the case where the input array itself is logically empty
    if input.len() == 0 && dim.is_none() {
        return Err(ArgmaxError::EmptyInput);
    }

    match dim {
        // --- Case 1: Flattened argmax (dim is None) ---
        None => {
            let (max_idx, _) = input
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less))
                .ok_or(ArgmaxError::EmptyInput)?;

            // Return a 0-dimensional array containing the flat index, cast to i64
            // Note: Potential panic if max_idx > i64::MAX on 32-bit systems (highly unlikely)
            Ok(arr0(max_idx as i64).into_dyn()) // <-- Cast usize to i64 here
        }

        // --- Case 2: Argmax along a specific axis (dim is Some) ---
        Some(axis_idx) => {
            let ndim = input.ndim();
            if axis_idx >= ndim {
                return Err(ArgmaxError::InvalidAxis(axis_idx));
            }

            let axis = Axis(axis_idx);
            let dim_size = input.shape()[axis_idx];

            if dim_size == 0 {
                return Err(ArgmaxError::ZeroDimSize(axis_idx));
            }

            // Use map_axis to apply a reduction along the specified axis.
            // The closure now returns i64.
            let result_no_keepdim: Array<i64, _> =
                input.map_axis(axis, |view: ArrayView<A, Ix1>| {
                    let (idx, _val) = view
                        .iter()
                        .copied()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less))
                        .unwrap(); // Safe due to dim_size > 0 check
                    // Note: Potential panic if idx > i64::MAX on 32-bit systems (highly unlikely)
                    idx as i64 // <-- Cast usize to i64 here
                });

            if keepdim {
                Ok(result_no_keepdim.insert_axis(axis).into_dyn())
            } else {
                Ok(result_no_keepdim.into_dyn())
            }
        }
    }
}
