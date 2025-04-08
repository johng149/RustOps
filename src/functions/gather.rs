use ndarray::{Array, ArrayD, Axis, Dimension, IxDyn, NdIndex};
use num_traits::{NumCast, Zero};
use std::fmt::Debug;
use std::mem::MaybeUninit;
use thiserror::Error; // Add thiserror for convenient error types

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum GatherError {
    #[error(
        "Input and index arrays must have the same number of dimensions (input: {input_ndim}, index: {index_ndim})"
    )]
    DimensionMismatch {
        input_ndim: usize,
        index_ndim: usize,
    },

    #[error("Dimension index {dim} is out of bounds for array with {ndim} dimensions")]
    InvalidDimension { dim: isize, ndim: usize },

    #[error(
        "Size of input ({input_size}) is less than size of index ({index_size}) along non-gathering dimension {axis}"
    )]
    ShapeMismatch {
        axis: usize,
        input_size: usize,
        index_size: usize,
    },

    #[error(
        "Index value {index_value:?} at {coords:?} is out of bounds for dimension {dim} with size {dim_size}"
    )]
    IndexOutOfBounds {
        coords: Vec<usize>,  // Store coords as Vec for ownership
        index_value: String, // Store problematic value as string for flexibility
        dim: usize,
        dim_size: usize,
    },

    #[error("Cannot cast index value {index_value:?} at {coords:?} to usize")]
    IndexCastError {
        coords: Vec<usize>,
        index_value: String,
    },

    #[error(
        "Negative index value {index_value:?} found at {coords:?}, which is not supported by gather"
    )]
    NegativeIndex {
        coords: Vec<usize>,
        index_value: String,
    },

    #[error("Internal error during array creation: {0}")]
    InternalShapeError(String), // For errors from Array::from_shape_vec etc.
}

/// Gathers values along a specified dimension `dim` using an index array.
///
/// Mimics the behavior of `torch.gather(input, dim, index)`.
/// The output array will have the same shape as the `index` array.
///
/// # Arguments
///
/// * `input`: The source array (`ArrayD<T>`).
/// * `dim`: The dimension along which to gather (`isize`). Negative values wrap around.
/// * `index`: The array containing the indices (`ArrayD<Ix>`) to gather from `input`.
///   The values in `index` must be valid indices for `input` along the `dim` dimension.
///
/// # Type Parameters
///
/// * `T`: The data type of the input and output arrays. Must implement `Clone` and `Debug`.
/// * `Ix`: The integer type of the indices in the `index` array. Must implement
///   `NdIndex` (automatically satisfied for integer types), `NumCast` (to convert to `usize`),
///   `PartialOrd` and `Zero` (for negativity check), `Copy`, and `Debug`.
///
/// # Returns
///
/// A `Result` containing:
/// * `Ok(ArrayD<T>)`: The resulting array with values gathered from `input`, having the same shape as `index`.
/// * `Err(GatherError)`: An error indicating why the gather operation failed (e.g., shape mismatch, index out of bounds).
pub fn gather<T, Ix>(
    input: &ArrayD<T>,
    dim: isize,
    index: &ArrayD<Ix>,
) -> Result<ArrayD<T>, GatherError>
where
    T: Clone + Debug,
    Ix: NumCast + PartialOrd + Zero + Copy + Debug, // Corrected: Removed NdIndex constraint for Ix
{
    // ... (function body remains the same) ...
    let input_ndim = input.ndim();
    let index_ndim = index.ndim();

    if input_ndim != index_ndim {
        return Err(GatherError::DimensionMismatch {
            input_ndim,
            index_ndim,
        });
    }

    // Handle 0-dimensional case separately (result is scalar if index is scalar)
    if input_ndim == 0 {
        if index.shape().len() == 0 {
            // Both are scalar. Index value must be 0.
            let index_val = index.first().copied().ok_or_else(|| {
                GatherError::InternalShapeError("Cannot get scalar index value".to_string())
            })?;
            // Check negativity first
            if index_val < Ix::zero() {
                return Err(GatherError::NegativeIndex {
                    coords: vec![],
                    index_value: format!("{:?}", index_val),
                });
            }
            // Try casting to usize
            let index_usize: usize = match NumCast::from(index_val) {
                Some(idx) => idx,
                None => {
                    return Err(GatherError::IndexCastError {
                        coords: vec![],
                        index_value: format!("{:?}", index_val),
                    });
                }
            };
            // Check if index (as usize) is 0
            if index_usize != 0 {
                return Err(GatherError::IndexOutOfBounds {
                    coords: vec![],
                    index_value: format!("{:?}", index_val),
                    dim: 0,
                    dim_size: 1, // Effective dim size for scalar is 1
                });
            }
            // Gather the single element from input
            let val = input.first().cloned().ok_or_else(|| {
                GatherError::InternalShapeError("Cannot get scalar input value".to_string())
            })?;
            return Ok(Array::from_elem(IxDyn(&[]), val)); // Return scalar array
        } else {
            // Input is scalar, index is not. This is likely an error according to gather rules.
            return Err(GatherError::DimensionMismatch {
                input_ndim: 0,
                index_ndim,
            });
        }
    }

    // --- Validate and normalize dimension ---
    let dim_usize = if dim >= 0 {
        let d = dim as usize;
        if d >= input_ndim {
            return Err(GatherError::InvalidDimension {
                dim,
                ndim: input_ndim,
            });
        }
        d
    } else {
        // Handle negative dim
        let resolved_dim = input_ndim as isize + dim;
        if resolved_dim < 0 {
            return Err(GatherError::InvalidDimension {
                dim,
                ndim: input_ndim,
            });
        }
        resolved_dim as usize
    };

    // --- Validate shapes ---
    let input_shape = input.shape();
    let index_shape = index.shape();

    for k in 0..input_ndim {
        if k != dim_usize && input_shape[k] < index_shape[k] {
            return Err(GatherError::ShapeMismatch {
                axis: k,
                input_size: input_shape[k],
                index_size: index_shape[k],
            });
        }
    }

    let input_dim_size = input_shape[dim_usize];
    // If the dimension we gather along is size 0, we can only succeed if the index
    // tensor is also empty, otherwise any index lookup is out of bounds.
    if input_dim_size == 0 && index.len() > 0 {
        // Find the first index element to report the error
        let (first_coords, first_index_val) = index.indexed_iter().next().unwrap(); // Safe because index.len() > 0
        return Err(GatherError::IndexOutOfBounds {
            coords: first_coords.slice().to_vec(),
            index_value: format!("{:?}", first_index_val),
            dim: dim_usize,
            dim_size: 0,
        });
    }

    // --- Prepare output array (same shape as index) ---
    // Use MaybeUninit for efficiency: allocate uninitialized memory and fill it.
    let output_shape = index.raw_dim().clone(); // Use raw_dim for Dimension trait bounds
    let mut output: Array<MaybeUninit<T>, _> = Array::uninit(output_shape);

    // Reusable buffer for constructing input coordinates
    let mut input_coords_buffer = vec![0; input_ndim];

    // --- Iterate and Gather ---
    // Iterate through the indices of the *output* array (which match the `index` array)
    for (idx_coords, output_elem_uninit) in output.indexed_iter_mut() {
        // Get the slice representation *first*. This borrows from idx_coords.
        let current_coords_slice: &[usize] = idx_coords.slice();

        // 1. Get the index value from the `index` array using the slice
        //    Pass the slice (&[usize]) to .get(). It implements NdIndex.
        let gather_idx_val = match index.get(current_coords_slice) {
            Some(val) => *val, // Dereference to get Ix, needs Ix: Copy
            None => {
                panic!("Internal error: Mismatch between output iterator and index array bounds.")
            }
        };

        // 2. Check for negative index, use the slice for error reporting
        if gather_idx_val < Ix::zero() {
            return Err(GatherError::NegativeIndex {
                coords: current_coords_slice.to_vec(), // Clone slice into Vec
                index_value: format!("{:?}", gather_idx_val),
            });
        }

        // 3. Cast the index value to usize for array indexing, use slice for error
        let gather_idx_usize: usize = match NumCast::from(gather_idx_val) {
            Some(idx) => idx,
            None => {
                return Err(GatherError::IndexCastError {
                    coords: current_coords_slice.to_vec(), // Clone slice into Vec
                    index_value: format!("{:?}", gather_idx_val),
                });
            }
        };

        // 4. Check if the index is within the bounds, use slice for error
        if gather_idx_usize >= input_dim_size {
            return Err(GatherError::IndexOutOfBounds {
                coords: current_coords_slice.to_vec(), // Clone slice into Vec
                index_value: format!("{:?}", gather_idx_val), // Report original index value
                dim: dim_usize,
                dim_size: input_dim_size,
            });
        }

        // 5. Construct the coordinates to access the `input` array using the slice
        // Check length just in case (should match input_ndim)
        if current_coords_slice.len() != input_ndim {
            panic!(
                "Internal error: Coordinate slice length mismatch. Expected {}, got {}. Coords: {:?}",
                input_ndim,
                current_coords_slice.len(),
                current_coords_slice
            );
        }
        // Copy current multi-dimensional index from the slice
        input_coords_buffer.copy_from_slice(current_coords_slice);
        // Replace the coordinate at `dim_usize` with the gathered index
        input_coords_buffer[dim_usize] = gather_idx_usize;

        // 6. Get the value from the `input` array using the buffer (which is &[usize])
        let value = match input.get(&input_coords_buffer[..]) {
            // Pass slice from buffer
            Some(v) => v.clone(), // Clone the value from input
            None => {
                panic!(
                    "Internal error: Failed to get value from input at {:?} despite bounds checks.",
                    input_coords_buffer
                );
            }
        };

        // 7. Write the gathered value to the output array
        *output_elem_uninit = MaybeUninit::new(value);
    }

    // --- Finalize Output ---
    // SAFETY: We iterated through every element using `indexed_iter_mut` and wrote a value
    // using `MaybeUninit::new` exactly once for each element.
    let final_output = unsafe { output.assume_init() };

    Ok(final_output)
}
