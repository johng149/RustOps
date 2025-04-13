use std::fmt::Debug;

use ndarray::{Array, ArrayD, Dimension, IxDyn, RemoveAxis};

use super::argmax::argmax;
use super::gather::gather;
use super::scatter::scatter;
use super::zeros_like::zeros_like;

pub fn maxi<T, D>(x: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float + Clone + Debug,
    D: Dimension + RemoveAxis,
{
    let dim: Option<usize> = Some(x.ndim() - 1);
    let max_indices = argmax(&x, dim, true).unwrap();
    let blank = zeros_like(x);
    let gather_dim: isize = dim.unwrap() as isize;
    let blank_scatter_dim = blank.ndim() as isize - 1;

    // Convert x to ArrayD before calling gather
    let x_arrayd: ArrayD<T> = x.to_owned().into_dimensionality().unwrap();

    // Now pass the ArrayD version to gather
    let values = gather(&x_arrayd, gather_dim, &max_indices).unwrap();

    // Change this to ArrayD<T> instead of ArrayD<f32>
    let mut blank_arrayd: ArrayD<T> = blank.to_owned().into_dimensionality().unwrap();

    let status = scatter(&mut blank_arrayd, blank_scatter_dim, &max_indices, &values);

    // Finally, convert back to the original dimension type and return
    blank_arrayd.into_dimensionality::<D>().unwrap()
}
