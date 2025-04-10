use ndarray::{ArrayD, Axis, IxDyn};
use std::cmp::Ordering;

/// Sorts each slice along the last dimension of the ArrayD in place.
/// Uses a fallback method (copying to Vec) if direct slice sorting fails.
///
/// # Arguments
///
/// * `arr` - A mutable reference to the `ArrayD` to be sorted.
///
/// # Type Parameters
///
/// * `A` - The element type of the array. Must implement `PartialOrd` and `Clone`.
///
/// # Panics
///
/// Panics if the array is 0-dimensional.
/// Panics if `partial_cmp` returns `None` during sorting (e.g., comparing NaN values
/// without a specific NaN handling strategy). The current implementation treats
/// elements causing `None` as equal for sorting purposes.
pub fn sort_last_dim<A>(arr: &mut ArrayD<A>)
where
    A: PartialOrd + Clone,
{
    let ndim = arr.ndim();
    if ndim == 0 {
        eprintln!("Warning: Cannot sort a 0-dimensional array.");
        return;
    }

    // We want to iterate over all "rows" (subarrays) where each row
    // is the collection of elements along the last axis
    let last_axis_index = ndim - 1;

    // This is the key change: we iterate over all axes EXCEPT the last one
    let mut outer_indices = Vec::new();
    for ax in 0..last_axis_index {
        outer_indices.push(Axis(ax));
    }

    // Using lanes_mut to get mutable views of each "row" along the last axis
    for mut lane in arr.lanes_mut(Axis(last_axis_index)) {
        // Each lane is a contiguous view along the last axis
        if let Some(slice) = lane.as_slice_mut() {
            // Sort the slice using partial_cmp
            slice.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        } else {
            // Fallback for non-contiguous views
            eprintln!(
                "Warning: Could not get mutable slice for lane (shape={:?}, strides={:?}). Using slower copy-based sort.",
                lane.shape(),
                lane.strides()
            );

            let mut temp_vec: Vec<A> = lane.iter().cloned().collect();
            temp_vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            for (view_elem, sorted_elem) in lane.iter_mut().zip(temp_vec.into_iter()) {
                *view_elem = sorted_elem;
            }
        }
    }
}
