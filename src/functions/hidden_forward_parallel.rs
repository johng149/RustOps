use super::expand::expand_at_dim;
use super::maxi::maxi;
use super::reduce::{self, reduce};
use super::reshape::reshape;
use super::sqrt::sqrt_ndarray;
use super::unsqueeze;
use super::{einsum::einsum_ndarray_dyn, unsqueeze::unsqueeze};
use ndarray::{ArrayD, NdFloat};
use num_traits::Float;

pub fn hidden_forward_parallel<T>(
    hidden_mm: ArrayD<T>,
    children_x: ArrayD<T>,
    rho: f32,
) -> ArrayD<T>
where
    T: Clone + std::fmt::Debug + ndarray::ScalarOperand + Float + NdFloat,
{
    let children_x_shape = children_x.shape();
    let (batch_size, total_children, children_mem_cols) = (
        children_x_shape[0],
        children_x_shape[1],
        children_x_shape[2],
    );

    let mm = if hidden_mm.ndim() == 4 {
        let unsqueezed = unsqueeze(&hidden_mm, 0);
        expand_at_dim(&unsqueezed, 0, batch_size).unwrap()
    } else {
        hidden_mm // batch_size, num_hidden_nodes, children_per_hidden, hidden_mems, children_mem_cols
    };

    let mm_shape = mm.shape();
    let (num_hidden_nodes, children_per_hidden, hidden_mems) =
        (mm_shape[1], mm_shape[2], mm_shape[3]);

    let x = maxi(&children_x);
    let x = reshape(
        &x,
        &[
            batch_size as i64,
            num_hidden_nodes as i64,
            children_per_hidden as i64,
            children_mem_cols as i64,
        ],
    )
    .unwrap();
    // we want the equation "batch hidden children h_mems c_mems, batch hidden children c_mems -> batch hidden h_mems"
    // however since `children` and `c_mems` both start with "c", we'll use `k` for `c_mems` and `c` for `children`,
    // similarly for `hidden` and `h_mems`, we'll use `j` for `h_mems` and `h` for `hidden`
    let propagation = einsum_ndarray_dyn("bhcjk,bhck->bhj", &[&mm, &x]).unwrap();
    let x_squared = x.mapv(|value| value * value);
    let reduced_x = reduce(&x_squared, "bhck,bhck->bh").unwrap();
    let x_norm = sqrt_ndarray(&reduced_x);

    let x_norm = unsqueeze(&x_norm, x_norm.ndim());

    let scaled_norm = x_norm.mapv(|value| value * T::from(children_per_hidden).unwrap());
    let transposed_norm = scaled_norm + T::from(rho).unwrap();
    let inversed_norm = transposed_norm.mapv(|value| T::one() / value);

    propagation * inversed_norm
}
