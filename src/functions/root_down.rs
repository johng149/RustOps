use super::growth_argmaxi::growth_argmaxi;
use ndarray::{ArrayD, NdFloat, ScalarOperand};
use num_traits::{Float, NumCast, PrimInt, ToPrimitive, Zero};
use std::fmt::Debug;

/// Applies growth_argmaxi to the last elements of upwards activations and layer counts.
/// This function does not mutate its inputs.
///
/// # Arguments
/// * `upwards` - Vector of activation tensors, the last one is used.
/// * `layer_counts` - Vector of count tensors, the last one is used.
/// * `eps` - Epsilon value for growth_argmaxi.
/// * `growth_threshold` - Threshold value for growth_argmaxi.
/// * `mark` - Marker value for reserved indices in growth_argmaxi.
///
/// # Returns
/// * A tuple containing:
///     * The resulting activation tensor after applying growth_argmaxi.
///     * The updated counts tensor resulting from growth_argmaxi.
pub fn root_down<T>(
    upwards: &Vec<ArrayD<T>>,
    layer_counts: &Vec<ArrayD<T>>,
    eps: T,
    growth_threshold: T,
    mark: i64,
) -> (ArrayD<T>, ArrayD<T>)
where
    T: Float + NumCast + ToPrimitive + Debug + Clone + ScalarOperand + NdFloat,
{
    // Get the last activation tensor
    let root_h_sub_l = upwards.last().expect("upwards vector should not be empty");

    // Get a reference to the last counts tensor
    let root_counts_ref = layer_counts
        .last()
        .expect("layer_counts vector should not be empty");

    // Call growth_argmaxi
    // growth_argmaxi takes the counts by reference, so no clone is needed here if its signature is &ArrayD<U>
    // If growth_argmaxi takes ownership (ArrayD<U>), a clone would be needed: root_counts_ref.clone()
    let (root_h_sub_l_star, updated_root_counts) = growth_argmaxi(
        root_h_sub_l,
        root_counts_ref, // Pass the counts by reference
        eps,
        growth_threshold,
        mark,
    );

    // Return the result from growth_argmaxi and the updated counts
    (root_h_sub_l_star, updated_root_counts)
}
