use ndarray::{ArrayD, NdFloat, ScalarOperand};
use num_traits::{Float, NumCast, ToPrimitive};
use std::borrow::Cow;
use std::fmt::Debug;

use super::einsum::einsum_ndarray_dyn;
use super::einsum_specialized;
use super::expand::expand_at_dim;
use super::reshape::reshape;
use super::unsqueeze::unsqueeze;

pub fn down_prop_parallel<T>(
    parent_h: &ArrayD<T>,
    parent_mm: &ArrayD<T>,
    child_h: &ArrayD<T>,
    coef: T,
) -> ArrayD<T>
where
    T: Float + NumCast + ToPrimitive + Debug + Clone + ScalarOperand + NdFloat,
{
    let child_h_shape = child_h.shape();
    let (batch_size, children, _children_dim) =
        (child_h_shape[0], child_h_shape[1], child_h_shape[2]);

    // Use Cow to avoid cloning parent_mm if it already has the correct dimensions
    let mm_cow = if parent_mm.ndim() == 4 {
        let unsqueezed = unsqueeze(parent_mm, 0);
        // let expanded = expand_at_dim(&unsqueezed, 0, batch_size)
        //     .expect("Failed to expand parent_mm at dimension 0");
        // Cow::Owned(expanded)
        Cow::Owned(unsqueezed)
    } else {
        Cow::Borrowed(parent_mm)
    };
    // Get a reference to the (potentially expanded) mm array
    let mm = mm_cow.as_ref();

    let mm_shape = mm.shape();
    let (_parent_nodes, children_per_parent, _parent_dim) = (mm_shape[1], mm_shape[2], mm_shape[3]);

    // Unsqueeze and expand parent_h
    let unsqueezed_parent_h = unsqueeze(parent_h, parent_h.ndim() - 1);
    // let argmaxi_parent_h = expand_at_dim(&unsqueezed_parent_h, 2, children_per_parent)
    //     .expect("Failed to expand parent_h at dimension 2");
    let argmaxi_parent_h = unsqueezed_parent_h;
    // argmaxi_parent_h is now an owned ArrayD<T>

    // Calculate original component scaled by coef
    // The multiplication `&ArrayD * T` returns an owned ArrayD<T>
    let orig = child_h * T::from(coef).unwrap();

    // Calculate new component using einsum
    // we want to do "batch parents children pdim, batch parents children pdim cdim -> batch parents children cdim"
    // however, `parents` and `pdim` has the same first letter, so we'll use `p` for `parents` and use
    // `d` for `pdim`. Similarly, `children` and `cdim` has the same first letter, so we'll use `c` for `children` and use
    // `k` for `cdim`.
    // einsum takes references to the input arrays
    // let new_einsum = einsum_ndarray_dyn("bpcd,bpcdk->bpck", &[&argmaxi_parent_h, mm])
    //     .expect("Einsum operation failed");
    let new_einsum =
        einsum_specialized::einsum_bpcd_bpcdk_bpck_dyn(&argmaxi_parent_h.view(), &mm.view())
            .unwrap();
    // new_einsum is an owned ArrayD<T>

    // Scale the new component and reshape
    let new_scaled = new_einsum * (T::from(1.0).unwrap() - coef);
    let new = reshape(&new_scaled, &[batch_size as i64, children as i64, -1])
        .expect("Reshape operation failed");
    // new is an owned ArrayD<T>

    // Combine original and new components
    // The addition `ArrayD + ArrayD` returns an owned ArrayD<T>
    orig + new
}
