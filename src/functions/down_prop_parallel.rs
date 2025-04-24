use ndarray::{ArrayD, NdFloat, ScalarOperand};
use num_traits::{Float, NumCast, ToPrimitive};
use std::fmt::Debug;

use super::einsum::einsum_ndarray_dyn;
use super::expand::expand_at_dim;
use super::reshape::reshape;
use super::unsqueeze::unsqueeze;

pub fn down_prop_parallel<T>(
    parent_h: ArrayD<T>,
    parent_mm: ArrayD<T>,
    child_h: ArrayD<T>,
    coef: f32,
) -> ArrayD<T>
where
    T: Float + NumCast + ToPrimitive + Debug + Clone + ScalarOperand + NdFloat,
{
    let child_h_shape = child_h.shape();
    let (batch_size, children, children_dim) =
        (child_h_shape[0], child_h_shape[1], child_h_shape[2]);

    let mm = if parent_mm.ndim() == 4 {
        let unsqueezed = unsqueeze(&parent_mm, 0);
        expand_at_dim(&unsqueezed, 0, batch_size).unwrap()
    } else {
        parent_mm
    };

    let mm_shape = mm.shape();
    let (parent_nodes, children_per_parent, parent_dim) = (mm_shape[1], mm_shape[2], mm_shape[3]);

    let argmaxi_parent_h = unsqueeze(&parent_h, parent_h.ndim() - 1);
    let argmaxi_parent_h = expand_at_dim(&argmaxi_parent_h, 2, children_per_parent).unwrap();
    let orig = child_h * T::from(coef).unwrap();

    // we want to do "batch parents children pdim, batch parents children pdim cdim -> batch parents children cdim"
    // however, `parents` and `pdim` has the same first letter, so we'll use `p` for `parents` and use
    // `d` for `pdim`. Similarly, `children` and `cdim` has the same first letter, so we'll use `c` for `children` and use
    // `k` for `cdim`.
    let new = einsum_ndarray_dyn("bpcd,bpcdk->bpck", &[&argmaxi_parent_h, &mm]).unwrap();
    let new = new * T::from(1.0 - coef).unwrap();
    let new = reshape(&new, &[batch_size as i64, children as i64, -1]).unwrap();

    orig + new
}
