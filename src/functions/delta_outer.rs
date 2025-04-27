use super::mem_delta::mem_delta;
use ndarray::{ArrayD, NdFloat};

/// Calculates the delta for the outer layer using the memory delta function.
/// This function does not mutate its inputs.
///
/// # Arguments
/// * `downwards` - Vector of downward activation tensors, the last one is used.
/// * `outer_mm` - The memory matrix for the outer layer.
/// * `sensory_input` - The sensory input tensor.
///
/// # Returns
/// * The resulting delta tensor for the outer layer.
pub fn delta_outer<T>(
    downwards: &Vec<ArrayD<T>>,
    outer_mm: &Vec<ArrayD<T>>,
    sensory_input: &ArrayD<T>,
) -> ArrayD<T>
where
    T: NdFloat,
{
    // Get the last downward activation tensor
    let outer_h_sub_l_star = &downwards
        .last()
        .expect("downwards vector should not be empty"); // Clone if mem_delta needs ownership or mutability, or pass by ref if possible
    let mm = &outer_mm
        .first()
        .expect("outer_mm vector should not be empty"); // Clone if mem_delta needs ownership or mutability, or pass by ref if possible

    // Call mem_delta
    let outer_delta = mem_delta(outer_h_sub_l_star, mm, sensory_input);

    // Return the result from mem_delta
    outer_delta
}
