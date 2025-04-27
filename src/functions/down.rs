use super::down_prop_parallel::{self, down_prop_parallel};
use super::growth_argmaxi::growth_argmaxi;
use super::root_down::root_down;
use ndarray::{ArrayD, NdFloat, ScalarOperand};
use num_traits::{Float, NumCast, PrimInt, ToPrimitive, Zero};
use std::fmt::Debug;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/src/functions/down.rs

/// Computes the downward pass through the layers.
///
/// Corresponds to the `down` method in the Python reference implementation.
/// This function does not mutate its inputs. It returns the updated counts along with the downward activations.
///
/// # Arguments
/// * `upwards` - A vector containing the activation tensors from the upward pass.
/// * `layer_counts` - A vector containing the count tensors for each layer.
/// * `layers` - A slice containing the memory tensors for each layer.
/// * `eps` - Epsilon value for `growth_argmaxi`.
/// * `coeff` - Coefficient for `down_prop_parallel`.
/// * `growth_threshold` - Threshold value for `growth_argmaxi`.
/// * `mark` - Marker value for reserved indices in `growth_argmaxi`.
///
/// # Returns
/// A tuple containing:
///   - A vector containing the downward activation tensors for each layer.
///     The order matches the Python implementation (root layer's result first).
///   - A vector containing the updated count tensors for each layer.
///
/// # Panics
/// Panics if `upwards`, `layer_counts`, or `layers` are empty or have inconsistent lengths.
pub fn down<T, U>(
    upwards: &Vec<ArrayD<T>>,
    layer_counts: &Vec<ArrayD<U>>,
    layers: &[ArrayD<T>],
    eps: T,
    coeff: T,
    growth_threshold: T,
    mark: i64,
) -> (Vec<ArrayD<T>>, Vec<ArrayD<U>>)
where
    T: Float + NumCast + ToPrimitive + Debug + Clone + ScalarOperand + NdFloat,
    U: Clone + ToPrimitive + Zero + Debug + PrimInt + ScalarOperand,
{
    let num_layers = layers.len();
    if upwards.is_empty() || layer_counts.is_empty() || num_layers == 0 {
        panic!("Input vectors/slices cannot be empty.");
    }
    if !(upwards.len() == num_layers && layer_counts.len() == num_layers) {
        panic!("Input vector/slice lengths must be consistent.");
    }

    // Create vector to hold new layer counts
    let mut new_layer_counts: Vec<ArrayD<U>> = Vec::with_capacity(num_layers);

    // Create vector to hold new activations
    let mut downwards: Vec<ArrayD<T>> = Vec::with_capacity(num_layers);

    // the first layer uses root_down
    let (root_h_sub_l_star, updated_root_counts) =
        root_down(&upwards, &layer_counts, eps, growth_threshold, mark);

    // Append the updated counts to the new_layer_counts vector
    new_layer_counts.push(updated_root_counts);
    // Append the new activations to the downwards vector
    downwards.push(root_h_sub_l_star);

    // Iterate through the remaining layers, starting from the second-to-last layer down to the first layer.
    // This corresponds to Python's range(len(upwards)-2, -1, -1)
    for i in (0..num_layers - 1).rev() {
        let h_sub_l = &upwards[i];
        let counts = &layer_counts[i];
        let layer = &layers[i + 1];
        let last_down = downwards
            .last()
            .expect("downwards vector should not be empty");
        let downed = down_prop_parallel(&last_down, &layer, &h_sub_l, coeff);
        let (h_sub_l_star, updated_counts) =
            growth_argmaxi(&downed, counts, eps, growth_threshold, mark);

        // Append the updated counts to the new_layer_counts vector
        new_layer_counts.push(updated_counts);
        // Append the new activations to the downwards vector
        downwards.push(h_sub_l_star);
    }

    // Return the collected results
    // because of the way we pushed the new_layer_counts, we need to reverse it
    new_layer_counts.reverse();
    (downwards, new_layer_counts)
}
