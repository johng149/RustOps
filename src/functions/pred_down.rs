use super::argmaxi::argmaxi;
use super::down_prop_parallel::down_prop_parallel;
use super::pred_root_down::pred_root_down;
use ndarray::{ArrayD, NdFloat, ScalarOperand};
use num_traits::{Float, NumCast, ToPrimitive};
use std::fmt::Debug;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/src/functions/pred_down.rs

/// Performs the downward prediction pass.
/// This function does not mutate its inputs.
///
/// # Arguments
/// * `upwards` - Vector of upward activation tensors.
/// * `layers` - Vector of layer weight tensors (e.g., `mm` matrices). Should have length `upwards.len() - 1`.
/// * `eps` - Epsilon value for argmaxi.
/// * `coeff` - Coefficient for down_prop_parallel.
///
/// # Returns
/// * A vector containing the downward predicted activations, starting from the root prediction.
pub fn pred_down<T>(upwards: &[ArrayD<T>], layers: &[ArrayD<T>], eps: T, coeff: T) -> Vec<ArrayD<T>>
where
    T: Float + NumCast + ToPrimitive + Debug + Clone + ScalarOperand + NdFloat,
{
    if upwards.is_empty() {
        return Vec::new();
    }
    if upwards.len() != layers.len() {
        panic!(
            "Mismatch between number of upward activations ({}) and layers ({})",
            upwards.len(),
            layers.len()
        );
    }

    let mut downwards = Vec::with_capacity(upwards.len());

    // Calculate the prediction for the root layer
    let root_pred = pred_root_down(&upwards.to_vec(), eps); // pred_root_down expects Vec
    downwards.push(root_pred);

    // Iterate downwards from the second-to-last layer to the first
    for i in (0..upwards.len() - 1).rev() {
        let h_sub_l = &upwards[i];
        let parent_h_star = downwards
            .last()
            .expect("Downwards vector should not be empty at this point");
        // layers[i] connects upwards[i] and upwards[i+1]
        // In the downward pass, we use layers[i+1] to propagate from downwards[-1] (related to upwards[i+1])
        // to the level of upwards[i].
        let mm = &layers[i + 1]; // Corresponds to self.layers[i+1] in python if layers index matches upward layer index it connects *to*

        let downed = down_prop_parallel(parent_h_star, mm, h_sub_l, coeff);
        let h_sub_l_star = argmaxi(&downed, eps);
        downwards.push(h_sub_l_star);
    }

    downwards
}
