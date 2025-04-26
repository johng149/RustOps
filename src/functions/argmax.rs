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

/// Finds the indices (as i64) of the **first** maximum values of an array along a given dimension.
/// Mimics the behavior of PyTorch's `torch.argmax` regarding duplicate maximum values (returns the first index).
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
/// * `Ok(Array<i64, IxDyn>)`: An array containing the indices (as `i64`) of the first maximum values.
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
/// differ slightly from PyTorch's `argmax`, which has specific NaN propagation/handling rules.
/// This implementation will typically *not* select a NaN as the maximum unless it's the first
/// element encountered and no larger non-NaN value is found later. If a comparison returns `None`
/// (due to NaN), the existing maximum is kept.
pub fn argmax<A, S, D>(
    input: &ArrayBase<S, D>,
    dim: Option<usize>,
    keepdim: bool,
) -> Result<Array<i64, IxDyn>, ArgmaxError>
where
    A: PartialOrd + Copy,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    // --- Helper function to find the index of the first max in an iterator ---
    // Returns Option<(index, value)>
    fn find_first_max<A: PartialOrd + Copy>(
        iter: impl Iterator<Item = (usize, A)>,
    ) -> Option<(usize, A)> {
        iter.fold(None, |acc, (current_index, current_value)| {
            match acc {
                // If this is the first element we've seen, it's the best so far
                None => Some((current_index, current_value)),
                // We've seen elements before, compare current with the best found so far
                Some((best_index_so_far, best_value_so_far)) => {
                    // Use partial_cmp for comparison
                    match current_value.partial_cmp(&best_value_so_far) {
                        // If current value is strictly greater, update the best
                        Some(Ordering::Greater) => Some((current_index, current_value)),
                        // If current value is less, equal, or comparison is indeterminate (NaN),
                        // keep the existing best (because we want the *first* max index).
                        Some(Ordering::Less) | Some(Ordering::Equal) | None => {
                            Some((best_index_so_far, best_value_so_far))
                        }
                    }
                }
            }
        })
    }

    // Handle the case where the input array itself is logically empty when finding overall max
    if input.len() == 0 && dim.is_none() {
        // Check explicitly because fold on empty iterator returns None, leading to the same error.
        // This check might be slightly redundant but clarifies intent.
        return Err(ArgmaxError::EmptyInput);
    }

    match dim {
        // --- Case 1: Flattened argmax (dim is None) ---
        None => {
            let first_max = find_first_max(input.iter().copied().enumerate());

            match first_max {
                Some((max_idx, _)) => {
                    // Return a 0-dimensional array containing the flat index, cast to i64
                    // Note: Potential panic if max_idx > i64::MAX on 32-bit systems (highly unlikely)
                    Ok(arr0(max_idx as i64).into_dyn()) // Cast usize to i64 here
                }
                // This case handles truly empty input arrays (input.len() == 0)
                None => Err(ArgmaxError::EmptyInput),
            }
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
                // Although map_axis might handle this, explicitly check for clarity
                // and to return the specific error type.
                return Err(ArgmaxError::ZeroDimSize(axis_idx));
            }

            // Use map_axis to apply a reduction along the specified axis.
            // The closure now returns i64.
            let result_no_keepdim: Array<i64, _> =
                input.map_axis(axis, |view: ArrayView<A, Ix1>| {
                    // Find the first maximum index within this 1D view
                    let (idx, _val) = find_first_max(view.iter().copied().enumerate())
                        .expect("Axis dimension size already checked to be non-zero"); // Safe due to dim_size > 0 check

                    // Note: Potential panic if idx > i64::MAX on 32-bit systems (highly unlikely)
                    idx as i64 // Cast usize to i64 here
                });

            if keepdim {
                // Use insert_axis to add the dimension back with size 1
                Ok(result_no_keepdim.insert_axis(axis).into_dyn())
            } else {
                // The dimension is already removed by map_axis
                Ok(result_no_keepdim.into_dyn())
            }
        }
    }
}
