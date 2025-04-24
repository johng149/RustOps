use ndarray::{ArrayD, NdFloat};
use num_traits::Float;

use super::einsum::{self, einsum_ndarray_dyn};
use super::expand::expand_at_dim;
use super::reduce::reduce;
use super::sqrt::sqrt_ndarray;
use super::unsqueeze::unsqueeze;

pub fn outer_forward_parallel<T>(mems: ArrayD<T>, xs: ArrayD<T>, rho: f32) -> ArrayD<T>
where
    T: Clone + std::fmt::Debug + ndarray::ScalarOperand + Float + NdFloat,
{
    let x_shape = xs.shape();
    let (batch_size, nodes, dim) = (x_shape[0], x_shape[1], x_shape[2]);

    let mems = mems.mapv(|value| value - T::from(0.5).unwrap());
    let m = if mems.ndim() != 4 {
        let unsqueezed = unsqueeze(&mems, 0);
        expand_at_dim(&unsqueezed, 0, batch_size).unwrap()
    } else {
        mems
    };
    let x = xs.mapv(|value| value - T::from(0.5).unwrap());

    let numerator = einsum_ndarray_dyn("bfmd,bfd->bfm", &[&m, &x]).unwrap();
    let numerator = numerator.mapv(|value| value * T::from(0.5).unwrap());
    let m_squared = m.mapv(|value| value * value);
    let x_squared = x.mapv(|value| value * value);
    let reduced_m = reduce(&m_squared, "bfmd,bfmd->bfm").unwrap();
    let m_norm = sqrt_ndarray(&reduced_m);
    let reduced_x = reduce(&x_squared, "bfd,bfd->bf").unwrap();
    let reduced_x_unsqueezed = unsqueeze(&reduced_x, reduced_x.ndim());
    let x_norm = sqrt_ndarray(&reduced_x_unsqueezed);

    let denominator = m_norm * x_norm;
    let denominator = denominator + T::from(rho).unwrap();
    let ratio = numerator / denominator;

    ratio + T::from(0.5).unwrap()
}
