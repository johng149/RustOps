use ndarray::prelude::*;
use ndarray::{Array, ArrayD, IxDyn, NdFloat};
use ndarray_einsum_beta::{ArrayLike, einsum}; // <-- Added LinalgScalar // <-- Use ArrayD explicitly

/// Performs Einstein summation for ndarray arrays.
///
/// This is a wrapper around the ndarray_einsum_beta crate's einsum function,
/// specifically tailored for ArrayD inputs.
///
/// # Arguments
///
/// * `equation` - A string describing the Einstein summation convention to use
/// * `tensors` - A slice of references to ArrayD<A> arrays to operate on
///
/// # Returns
///
/// The result of the Einstein summation as an ndarray ArrayD<A>
pub fn einsum_ndarray_dyn<'a, A>(
    // Renamed slightly for clarity
    equation: &str,
    tensors: &[&'a ArrayD<A>], // <-- Accept slice of &ArrayD<A>
) -> Result<ArrayD<A>, &'static str>
// <-- Return ArrayD<A> (alias for Array<A, IxDyn>)
where
    A: NdFloat, // <-- Add LinalgScalar bound
                // No generic D needed here anymore
{
    // should call with let tensors_to_sum = [&x, &y]; and then pass tensors_to_sum

    // 1. Create a Vec to hold the trait object references
    // 2. Iterate through the input slice `tensors`
    // 3. For each `&ArrayD<A>`, coerce it to `&dyn ArrayLike<A>`
    // 4. Collect these trait object references into the Vec
    let operands: Vec<&dyn ArrayLike<A>> = tensors
        .iter()
        .map(|&tensor| tensor as &dyn ArrayLike<A>) // Perform the coercion here
        .collect();

    // 5. Pass a slice of the Vec<&dyn ArrayLike<A>> to the einsum function
    einsum(equation, &operands)
}
