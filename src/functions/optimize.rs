use super::delta::delta;
use super::down::down;
use super::optim::optim;
use super::up::up;
use ndarray::{ArrayD, NdFloat, ScalarOperand};
use num_traits::{Float, FromPrimitive, NumCast, PrimInt, ToPrimitive, Zero};
use std::fmt::Debug;
use std::vec::Vec;

/// Performs a full optimization cycle: up pass, down pass, delta calculation, and optimization step.
///
/// This function orchestrates the calls to `up`, `down`, `delta`, and `optim`
/// based on the provided state and input. It corresponds to the `optimize`
/// method in the Python reference implementation.
///
/// # Arguments
/// * `layers` - Current memory tensors for each layer.
/// * `layer_counts` - Current count tensors for each layer.
/// * `sensory_input` - The input tensor for the current cycle.
/// * `t` - The current iteration count.
/// * `a` - Parameter 'a' used in growth threshold calculation within `optim`.
/// * `rho` - Regularization parameter used in the `up` pass.
/// * `eps` - Epsilon value used in the `down` pass (`growth_argmaxi`).
/// * `coeff` - Coefficient used in the `down` pass (`down_prop_parallel`).
/// * `growth_threshold` - Current growth threshold used in the `down` pass.
/// * `mark` - Marker value for reserved indices used in the `down` pass (`growth_argmaxi`).
///
/// # Returns
/// A tuple containing the updated state after one optimization cycle:
/// * `Vec<ArrayD<T>>` - The updated memory tensors for all layers.
/// * `Vec<ArrayD<U>>` - The updated count tensors for all layers.
/// * `usize` - The incremented iteration count (`t + 1`).
/// * `T` - The newly calculated growth threshold.
///
/// # Type Parameters
/// * `T`: The floating-point type used for activations and memory tensors. Must satisfy constraints
///        required by `up`, `down`, `delta`, and `optim`.
/// * `U`: The integer type used for count tensors. Must satisfy constraints required by `down` and `optim`.
///
/// # Panics
/// Panics if any of the underlying functions (`up`, `down`, `delta`, `optim`) panic,
/// for example due to empty inputs, mismatched dimensions, or numerical issues.
pub fn optimize<T, U>(
    layers: &[ArrayD<T>],
    layer_counts: &Vec<ArrayD<U>>,
    sensory_input: ArrayD<T>,
    t: usize,
    a: T,
    rho: f32, // `up` specifically uses f32 for rho
    eps: T,
    coeff: T,
    growth_threshold: T,
    mark: i64,
) -> (Vec<ArrayD<T>>, Vec<ArrayD<U>>, usize, T)
where
    T: NdFloat
        + Float
        + NumCast
        + ToPrimitive
        + Debug
        + Clone
        + ScalarOperand
        + Copy
        + FromPrimitive,
    U: Clone + ToPrimitive + Zero + Debug + PrimInt + ScalarOperand,
{
    // Clone sensory_input as both `up` and `delta` need it, and `up` takes ownership.
    let sensory_input_for_up = sensory_input.clone();
    let sensory_input_for_delta = sensory_input; // Original can be moved into delta

    // 1. Up pass
    // `up` takes layers by slice, sensory_input by value.
    let upwards = up(layers, sensory_input_for_up, rho);

    // 2. Down pass
    // `down` takes upwards and layer_counts by reference, layers by slice.
    // It returns the downward activations and the *updated* layer counts.
    let (downwards, new_layer_counts) = down(
        &upwards,
        layer_counts,
        layers,
        eps,
        coeff,
        growth_threshold,
        mark,
    );

    // 3. Delta calculation
    // `delta` takes downwards and layers by reference, sensory_input by value.
    // Note: `layers` here refers to the memory matrices *before* the optim step.
    let deltas = delta(&downwards, &layers.to_vec(), sensory_input_for_delta); // Pass original layers

    // currently `new_layer_counts` is a Vec<ArrayD<U>>, but it should be a Vec<ArrayD<T>> as that
    // is what `optim` expects, so we need to convert it.
    // This is a workaround for the type mismatch.
    let new_layer_counts_optim: Vec<ArrayD<T>> = new_layer_counts
        .iter()
        .map(|count| count.mapv(|x| T::from(x.to_i64().unwrap()).unwrap()))
        .collect();

    // 4. Optimization step
    // `optim` takes deltas, layers, and layer_counts by reference.
    // It uses the *original* layers and the *updated* counts from the down pass
    // to calculate the updates and return the *new* layers, *new* t, and *new* growth_threshold.
    let (updated_layers, new_t, new_growth_threshold) =
        optim(&deltas, &layers.to_vec(), &new_layer_counts_optim, t, a); // Pass original layers and new counts

    // Return the updated state
    (
        updated_layers,
        new_layer_counts,
        new_t,
        new_growth_threshold,
    )
}
