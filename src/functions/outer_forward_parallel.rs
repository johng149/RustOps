use super::einsum::{self, einsum_ndarray_dyn};
use super::einsum_specialized::einsum_bfmd_bfd_bfm_dyn;
// No longer needed if we rely on broadcasting:
// use super::expand::expand_at_dim;
use super::reduce::reduce;
use super::reduce_specialized;
use super::sqrt::sqrt_ndarray;
use super::unsqueeze::unsqueeze;
// Import ArrayViewD for working with views
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn, NdFloat};
use num_traits::Float;

pub fn outer_forward_parallel<T>(mems: ArrayD<T>, xs: ArrayD<T>, rho: f32) -> ArrayD<T>
where
    T: Clone + std::fmt::Debug + ndarray::ScalarOperand + Float + NdFloat,
{
    let x_shape = xs.shape();
    let (batch_size, nodes, dim) = (x_shape[0], x_shape[1], x_shape[2]);

    // Process mems: subtract 0.5. This might create an owned array anyway,
    // but we avoid the *additional* large allocation/copy from expand_at_dim.
    let mems_processed = mems.mapv(|value| value - T::from(0.5).unwrap());

    // Create a *view* `m_view` that has the correct dimensions for broadcasting.
    // We need temporary storage if unsqueezing is necessary, as unsqueeze returns owned.
    let mut unsqueezed_mems_storage: Option<ArrayD<T>> = None;
    let m_view: ArrayViewD<'_, T>; // The view we'll use in calculations

    if mems_processed.ndim() == 3 {
        // Need to unsqueeze to [1, nodes, mem_size, dim]
        // `unsqueeze` returns an owned ArrayD.
        let unsqueezed = unsqueeze(&mems_processed, 0);
        // Store it so the view remains valid.
        unsqueezed_mems_storage = Some(unsqueezed);
        // Create a view into the newly created unsqueezed array.
        m_view = unsqueezed_mems_storage.as_ref().unwrap().view().into_dyn();
        // m_view now has shape [1, nodes, mem_size, dim]
    } else if mems_processed.ndim() == 4 {
        // It's already 4D. Check if its batch dimension needs broadcasting.
        if mems_processed.shape()[0] == 1 && batch_size > 1 {
            // Shape is [1, nodes, mem_size, dim], broadcasting will handle it.
            m_view = mems_processed.view().into_dyn();
        } else if mems_processed.shape()[0] == batch_size {
            // Shape is [batch_size, nodes, mem_size, dim], no broadcasting needed on batch dim.
            m_view = mems_processed.view().into_dyn();
        } else {
            // Error: Mismatched batch dimensions and not broadcastable.
            panic!(
                "Input mems has 4 dimensions, but its first dimension ({}) does not match xs batch_size ({}) and is not 1 for broadcasting.",
                mems_processed.shape()[0],
                batch_size
            );
        }
    } else {
        panic!(
            "Input mems must have 3 or 4 dimensions, but has {} dimensions.",
            mems_processed.ndim()
        );
    }
    // After this block, m_view is a view with shape [1 or batch_size, nodes, mem_size, dim]

    // Process x: subtract 0.5
    let x = xs.mapv(|value| value - T::from(0.5).unwrap());
    let x_view = x.view(); // Shape [batch_size, nodes, dim]

    // --- Calculations using the view `m_view` ---

    // Einsum: Should handle broadcasting implicitly.
    // If m_view is [1, n, m, d] and x_view is [b, n, d], einsum treats m_view as [b, n, m, d].
    // let numerator = einsum_ndarray_dyn("bfmd,bfd->bfm", &[&m_view.into_dyn(), &x_view.into_dyn()]).unwrap();
    // Assuming your specialized function also handles views and broadcasting correctly:
    let numerator = einsum_bfmd_bfd_bfm_dyn(&m_view, &x_view).unwrap(); // Pass views
    let numerator = numerator.mapv(|value| value * T::from(0.5).unwrap()); // mapv allocates result

    // Square operations: mapv will likely create a full owned array if m_view is broadcasted,
    // but we avoided the initial explicit expand_at_dim copy.
    let m_squared = m_view.mapv(|value| value * value); // Result is owned, potentially broadcasted shape [b, n, m, d]
    let x_squared = x_view.mapv(|value| value * value); // Result is owned, shape [b, n, d]

    // Reductions should work fine on the (potentially broadcasted) m_squared view
    // let reduced_m = reduce(&m_squared, "bfmd,bfmd->bfm").unwrap();
    let reduced_m = reduce_specialized::reduce_bfmd_to_bfm(&m_squared.view()).unwrap(); // Pass view
    let m_norm = sqrt_ndarray(&reduced_m); // Shape [b, n, m]

    // let reduced_x = reduce(&x_squared, "bfd,bfd->bf").unwrap();
    let reduced_x = reduce_specialized::reduce_bfd_to_bf(&x_squared.view()).unwrap(); // Pass view, shape [b, n]

    // Unsqueeze reduced_x to [b, n, 1] for broadcasting with m_norm [b, n, m]
    let reduced_x_unsqueezed = unsqueeze(&reduced_x, reduced_x.ndim()); // Shape [b, n, 1]
    let x_norm = sqrt_ndarray(&reduced_x_unsqueezed); // Shape [b, n, 1]

    // Element-wise ops: Broadcasting handles [b, n, m] * [b, n, 1] -> [b, n, m]
    let denominator = &m_norm * &x_norm; // Use references to avoid move if m_norm/x_norm are needed later
    let denominator = denominator + T::from(rho).unwrap();
    let ratio = numerator / denominator; // numerator is [b, n, m]

    ratio + T::from(0.5).unwrap()
}
