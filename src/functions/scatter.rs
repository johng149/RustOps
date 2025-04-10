use ndarray::{Array, ArrayD, Axis, Dimension, IxDyn, NdIndex};
use num_traits::{NumCast, Zero};
use std::fmt::Debug;
// We don't need MaybeUninit here as we modify an existing array.
use thiserror::Error;

// Reusing GatherError for simplicity, but renaming to ScatterError
// and adjusting messages might be cleaner in a real library.
// For this example, we'll adapt GatherError slightly mentally,
// or create a new ScatterError enum. Let's create ScatterError for clarity.

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ScatterError {
    #[error(
        "Target, index, and source arrays must have the same number of dimensions (target: {target_ndim}, index: {index_ndim}, source: {source_ndim})"
    )]
    DimensionMismatch {
        target_ndim: usize,
        index_ndim: usize,
        source_ndim: usize,
    },

    #[error("Dimension index {dim} is out of bounds for array with {ndim} dimensions")]
    InvalidDimension { dim: isize, ndim: usize },

    #[error(
        "Index and source arrays must have the same shape (index: {index_shape:?}, source: {source_shape:?})"
    )]
    IndexSourceShapeMismatch {
        index_shape: Vec<usize>, // Store shapes for clarity
        source_shape: Vec<usize>,
    },

    #[error(
        "Size of target ({target_size}) must be >= size of index/source ({index_size}) along non-scattering dimension {axis}"
    )]
    TargetTooSmall {
        axis: usize,
        target_size: usize,
        index_size: usize, // Index and source have same size here
    },

    #[error(
        "Index value {index_value:?} at {coords:?} is out of bounds for dimension {dim} with size {dim_size} in the target array"
    )]
    IndexOutOfBounds {
        coords: Vec<usize>,
        index_value: String,
        dim: usize,
        dim_size: usize,
    },

    #[error("Cannot cast index value {index_value:?} at {coords:?} to usize")]
    IndexCastError {
        coords: Vec<usize>,
        index_value: String,
    },

    #[error(
        "Negative index value {index_value:?} found at {coords:?}, which is not supported by scatter"
    )]
    NegativeIndex {
        coords: Vec<usize>,
        index_value: String,
    },

    #[error("Internal error: {0}")]
    InternalError(String), // For unexpected logic errors
}

/// Scatters values from a `source` array into the `target` array along a specified
/// dimension `dim` using an `index` array.
///
/// Modifies the `target` array in place. Mimics the behavior of
/// `target.scatter_(dim, index, source)`.
///
/// The `index` and `source` arrays must have the same shape.
/// The output (modified `target`) will have the same shape as the original `target`.
///
/// # Arguments
///
/// * `target`: The array to scatter values into (`&mut ArrayD<T>`). Modified in place.
/// * `dim`: The dimension along which to scatter (`isize`). Negative values wrap around.
/// * `index`: The array containing the indices (`ArrayD<Ix>`) specifying where in `target`
///   to write the values from `source`.
///   The values in `index` must be valid indices for `target` along the `dim` dimension.
/// * `source`: The array containing the values (`ArrayD<T>`) to scatter into `target`.
///
/// # Type Parameters
///
/// * `T`: The data type of the target and source arrays. Must implement `Clone` and `Debug`.
/// * `Ix`: The integer type of the indices in the `index` array. Must implement
///   `NumCast` (to convert to `usize`), `PartialOrd` and `Zero` (for negativity check),
///   `Copy`, and `Debug`.
///
/// # Returns
///
/// A `Result` containing:
/// * `Ok(())`: Indicates the scatter operation completed successfully.
/// * `Err(ScatterError)`: An error indicating why the scatter operation failed.
pub fn scatter<T, Ix>(
    target: &mut ArrayD<T>,
    dim: isize,
    index: &ArrayD<Ix>,
    source: &ArrayD<T>,
) -> Result<(), ScatterError>
where
    T: Clone + Debug,
    Ix: NumCast + PartialOrd + Zero + Copy + Debug,
{
    let target_ndim = target.ndim();
    let index_ndim = index.ndim();
    let source_ndim = source.ndim();

    // --- Validate Dimensions ---
    if !(target_ndim == index_ndim && index_ndim == source_ndim) {
        return Err(ScatterError::DimensionMismatch {
            target_ndim,
            index_ndim,
            source_ndim,
        });
    }

    // --- Handle 0-dimensional case ---
    if target_ndim == 0 {
        // All must be scalar
        if index.shape().len() == 0 && source.shape().len() == 0 {
            let index_val = index.first().copied().ok_or_else(|| {
                ScatterError::InternalError("Cannot get scalar index value".to_string())
            })?;
            // Check negativity
            if index_val < Ix::zero() {
                return Err(ScatterError::NegativeIndex {
                    coords: vec![],
                    index_value: format!("{:?}", index_val),
                });
            }
            // Try casting to usize
            let index_usize: usize = match NumCast::from(index_val) {
                Some(idx) => idx,
                None => {
                    return Err(ScatterError::IndexCastError {
                        coords: vec![],
                        index_value: format!("{:?}", index_val),
                    });
                }
            };
            // Check bounds (must be 0 for scalar target)
            if index_usize != 0 {
                return Err(ScatterError::IndexOutOfBounds {
                    coords: vec![],
                    index_value: format!("{:?}", index_val),
                    dim: 0,
                    dim_size: 1, // Effective dim size for scalar target is 1
                });
            }
            // Get source value
            let source_val = source.first().cloned().ok_or_else(|| {
                ScatterError::InternalError("Cannot get scalar source value".to_string())
            })?;
            // Write to target
            if let Some(target_elem) = target.first_mut() {
                *target_elem = source_val;
                Ok(())
            } else {
                Err(ScatterError::InternalError(
                    "Cannot get mutable scalar target value".to_string(),
                ))
            }
        } else {
            // Mismatch in scalar/non-scalar
            Err(ScatterError::DimensionMismatch {
                target_ndim: 0,
                index_ndim,
                source_ndim,
            })
        }
    } else {
        // ndim > 0
        // --- Validate and normalize dimension ---
        let dim_usize = if dim >= 0 {
            let d = dim as usize;
            if d >= target_ndim {
                return Err(ScatterError::InvalidDimension {
                    dim,
                    ndim: target_ndim,
                });
            }
            d
        } else {
            // Handle negative dim
            let resolved_dim = target_ndim as isize + dim;
            if resolved_dim < 0 {
                return Err(ScatterError::InvalidDimension {
                    dim,
                    ndim: target_ndim,
                });
            }
            resolved_dim as usize
        };

        // --- Validate shapes ---
        let target_shape = target.shape();
        let index_shape = index.shape();
        let source_shape = source.shape();

        // 1. Index and Source shapes must match exactly
        if index_shape != source_shape {
            return Err(ScatterError::IndexSourceShapeMismatch {
                index_shape: index_shape.to_vec(),
                source_shape: source_shape.to_vec(),
            });
        }

        // 2. Target shape must be >= Index/Source shape along non-scattering dimensions
        for k in 0..target_ndim {
            if k != dim_usize && target_shape[k] < index_shape[k] {
                // index_shape[k] == source_shape[k] here
                return Err(ScatterError::TargetTooSmall {
                    axis: k,
                    target_size: target_shape[k],
                    index_size: index_shape[k],
                });
            }
        }

        let target_dim_size = target_shape[dim_usize];

        // If the target dimension we scatter along is size 0, we can only succeed if the
        // index tensor is also empty (which implies source is empty). Otherwise, any
        // index value >= 0 is out of bounds.
        if target_dim_size == 0 && index.len() > 0 {
            // Find the first index element to report the error
            let (first_coords, first_index_val) = index.indexed_iter().next().unwrap(); // Safe because index.len() > 0
            // Check negativity first for a more specific error if applicable
            if *first_index_val < Ix::zero() {
                return Err(ScatterError::NegativeIndex {
                    coords: first_coords.slice().to_vec(),
                    index_value: format!("{:?}", first_index_val),
                });
            }
            // Otherwise, it's out of bounds because dim_size is 0
            return Err(ScatterError::IndexOutOfBounds {
                coords: first_coords.slice().to_vec(),
                index_value: format!("{:?}", first_index_val),
                dim: dim_usize,
                dim_size: 0,
            });
        }

        // --- Prepare for iteration ---
        // Reusable buffer for constructing target coordinates
        let mut target_coords_buffer = vec![0; target_ndim];

        // --- Iterate and Scatter ---
        // Iterate through the indices of the `index` array (and `source` array)
        for (idx_coords, &index_val) in index.indexed_iter() {
            // Use `&index_val` as Ix: Copy
            // Get the slice representation for indexing and error reporting
            let current_coords_slice: &[usize] = idx_coords.slice();

            // 1. Check for negative index
            if index_val < Ix::zero() {
                return Err(ScatterError::NegativeIndex {
                    coords: current_coords_slice.to_vec(),
                    index_value: format!("{:?}", index_val),
                });
            }

            // 2. Cast the index value to usize
            let scatter_idx_usize: usize = match NumCast::from(index_val) {
                Some(idx) => idx,
                None => {
                    return Err(ScatterError::IndexCastError {
                        coords: current_coords_slice.to_vec(),
                        index_value: format!("{:?}", index_val),
                    });
                }
            };

            // 3. Check if the index is within the bounds of the target dimension
            if scatter_idx_usize >= target_dim_size {
                return Err(ScatterError::IndexOutOfBounds {
                    coords: current_coords_slice.to_vec(),
                    index_value: format!("{:?}", index_val), // Report original index value
                    dim: dim_usize,
                    dim_size: target_dim_size,
                });
            }

            // 4. Get the value from the `source` array at the current coordinates
            // We already checked that index and source shapes match.
            let source_value = match source.get(current_coords_slice) {
                Some(v) => v.clone(), // Clone the value from source
                None => {
                    // This should be unreachable if shape checks passed
                    return Err(ScatterError::InternalError(format!(
                        "Internal error: Failed to get value from source at {:?} despite shape checks.",
                        current_coords_slice
                    )));
                }
            };

            // 5. Construct the coordinates to access the `target` array
            // Check length just in case (should match target_ndim)
            if current_coords_slice.len() != target_ndim {
                return Err(ScatterError::InternalError(format!(
                    "Internal error: Coordinate slice length mismatch. Expected {}, got {}. Coords: {:?}",
                    target_ndim,
                    current_coords_slice.len(),
                    current_coords_slice
                )));
            }
            // Copy current multi-dimensional index from the slice
            target_coords_buffer.copy_from_slice(current_coords_slice);
            // Replace the coordinate at `dim_usize` with the scatter index
            target_coords_buffer[dim_usize] = scatter_idx_usize;

            // 6. Get a mutable reference to the element in the `target` array
            let target_elem_ref = match target.get_mut(&target_coords_buffer[..]) {
                // Pass slice from buffer
                Some(elem_ref) => elem_ref,
                None => {
                    // This should be unreachable if bounds checks passed
                    return Err(ScatterError::InternalError(format!(
                        "Internal error: Failed to get mutable reference from target at {:?} despite bounds checks.",
                        target_coords_buffer
                    )));
                }
            };

            // 7. Write the source value to the target array element
            *target_elem_ref = source_value;
        }

        Ok(()) // Scatter successful
    } // End ndim > 0 case
}
