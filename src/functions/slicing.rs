use ndarray::{Array, ArrayD, Axis, IxDyn, NdFloat, SliceInfo, SliceInfoElem};
use ndarray::{ShapeError, prelude::*};
use std::convert::TryFrom;

/// Slices a tensor like `[:, :, -1:]` for a 3D+ tensor.
///
/// Selects all elements along the first N-1 dimensions and the last element
/// along the last dimension, maintaining the dimensionality.
///
/// # Arguments
///
/// * `input` - The input ArrayD to slice. Must have at least 3 dimensions.
///
/// # Returns
///
/// A new owned ArrayD containing the slice, or an error if the input
/// tensor has fewer than 3 dimensions.
pub fn slice_last_dim<A>(input: &ArrayD<A>) -> Result<ArrayD<A>, &'static str>
where
    A: Clone,
{
    let ndim = input.ndim();
    // 1. Check dimensions -> return Err if invalid
    if ndim < 3 {
        return Err("Input tensor must have at least 3 dimensions for slice '[:, :, -1:]'");
    }

    // 2. Build slice_info_elems (handling edge cases like empty dims)
    let slice_info_elems = {
        let last_dim_size = input.shape()[ndim - 1];
        if last_dim_size == 0 {
            // Handle edge case of empty last dimension -> slice is 0..0
            let mut elems = vec![
                SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1
                };
                ndim
            ];
            elems[ndim - 1] = SliceInfoElem::Slice {
                start: 0,
                end: Some(0),
                step: 1,
            }; // Empty slice 0..0 is valid
            elems
        } else {
            // Last dim has elements -> slice is -1..
            let mut elems = Vec::with_capacity(ndim);
            for _ in 0..ndim - 1 {
                elems.push(SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                });
            }
            // start: -1 is valid since last_dim_size > 0
            elems.push(SliceInfoElem::Slice {
                start: -1,
                end: None,
                step: 1,
            });
            elems
        }
    };

    // 3. Try converting elems to SliceInfo, mapping error and using '?'
    let slice_info_dyn = SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info_elems.as_slice())
        .map_err(|_| "Internal error: Failed to create dynamic slice info")?; // This handles Result from try_from

    // 4. Call slice() - Panics if slice_info_dyn is invalid *despite* our checks (unlikely)
    //    No Result handling needed here as slice() doesn't return one.
    let view = input.slice(slice_info_dyn);

    // 5. Convert view to owned and return Ok
    Ok(view.to_owned())
}

/// Slices a tensor like `[:, :amount]` for a 2D+ tensor.
///
/// Selects all elements along the first dimension and the first `amount`
/// elements along the second dimension.
///
/// # Arguments
///
/// * `input` - The input ArrayD to slice. Must have at least 2 dimensions.
/// * `amount` - The number of elements to take from the second dimension.
///
/// # Returns
///
/// A new owned ArrayD containing the slice, or an error if the input tensor
/// has fewer than 2 dimensions or if `amount` is invalid.
pub fn slice_second_dim<A>(input: &ArrayD<A>, amount: usize) -> Result<ArrayD<A>, &'static str>
where
    A: Clone,
{
    let ndim = input.ndim();
    // 1. Check dimensions -> return Err if invalid
    if ndim < 2 {
        return Err("Input tensor must have at least 2 dimensions for slice '[:, :amount]'");
    }

    // Clamp amount to prevent out-of-bounds access in slice info
    let dim1_size = input.shape()[1];
    let effective_amount = std::cmp::min(amount, dim1_size);

    // 2. Build slice_info_elems
    let slice_info_elems = {
        let mut elems = Vec::with_capacity(ndim);
        elems.push(SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        }); // Dim 0: ..
        // Dim 1: 0..effective_amount. Valid because effective_amount <= dim1_size
        elems.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(effective_amount as isize),
            step: 1,
        });
        for _ in 2..ndim {
            // Remaining dims: ..
            elems.push(SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            });
        }
        elems
    };

    // 3. Try converting elems to SliceInfo, mapping error and using '?'
    let slice_info_dyn = SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info_elems.as_slice())
        .map_err(|_| "Internal error: Failed to create dynamic slice info")?; // Handles Result from try_from

    // 4. Call slice() - Panics if slice_info_dyn is invalid *despite* our checks (unlikely)
    //    No Result handling needed here.
    let view = input.slice(slice_info_dyn);

    // 5. Convert view to owned and return Ok
    Ok(view.to_owned())
}
