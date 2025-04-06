use ndarray::prelude::*;

/// Computes the absolute value of each element in a matrix for ndarray arrays.
pub fn abs_ndarray<T, D>(matrix: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: ndarray::Dimension,
{
    matrix.mapv(|x| x.abs())
}
