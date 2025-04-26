use super::argmaxi::argmaxi;
use ndarray::{ArrayD, Dimension, IxDyn, RemoveAxis, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// Applies argmaxi to the last element of the upwards activations.
/// This function does not mutate its inputs.
///
/// # Arguments
/// * `upwards` - Vector of activation tensors, the last one is used.
/// * `eps` - Epsilon value for argmaxi.
///
/// # Returns
/// * The resulting activation tensor after applying argmaxi.
pub fn pred_root_down<T>(upwards: &Vec<ArrayD<T>>, eps: T) -> ArrayD<T>
where
    T: Float + Clone + Debug + ScalarOperand,
{
    // Get the last activation tensor
    let root_h_sub_l = upwards.last().expect("upwards vector should not be empty");

    // Call argmaxi
    let root_h_sub_l_star = argmaxi(root_h_sub_l, eps);

    // Return the result from argmaxi
    root_h_sub_l_star
}
