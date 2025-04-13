use ndarray::Zip;
use ndarray::prelude::*;
use ndarray::{Array, ArrayD, Axis, IxDyn};
use ndarray::{Data, DataMut, Slice};
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

pub fn argsort_last_dim(arr: &ArrayD<f32>) -> ArrayD<usize> {
    // Get the shape of the input array
    let shape = arr.shape().to_vec();

    // If the array is empty, return an empty array
    if shape.is_empty() {
        return ArrayD::from_elem(IxDyn(&[]), 0);
    }

    // Create a result array with the same shape as the input
    let mut result = ArrayD::zeros(IxDyn(&shape));

    // Call the recursive helper function to fill the result array
    fill_result_recursive(arr, &mut result, &mut Vec::new(), &shape);

    result
}

fn fill_result_recursive(
    arr: &ArrayD<f32>,
    result: &mut ArrayD<usize>,
    indices: &mut Vec<usize>,
    shape: &[usize],
) {
    if indices.len() == shape.len() - 1 {
        // We've reached the penultimate dimension, now handle the last dimension
        let last_dim_size = shape.last().unwrap();

        // Create a vector of indices for the slice along the last dimension
        let mut indices_vec: Vec<usize> = (0..*last_dim_size).collect();

        // Sort the indices based on the values in the slice
        indices_vec.sort_by(|&i, &j| {
            let mut i_indices = indices.clone();
            i_indices.push(i);
            let mut j_indices = indices.clone();
            j_indices.push(j);

            let a = *arr.get(i_indices.as_slice()).unwrap();
            let b = *arr.get(j_indices.as_slice()).unwrap();

            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        });

        // Store the sorted indices in the result array
        for i in 0..*last_dim_size {
            let mut result_indices = indices.clone();
            result_indices.push(i);
            *result.get_mut(result_indices.as_slice()).unwrap() = indices_vec[i];
        }
    } else {
        // Recurse into the next dimension
        for i in 0..shape[indices.len()] {
            indices.push(i);
            fill_result_recursive(arr, result, indices, shape);
            indices.pop();
        }
    }
}
