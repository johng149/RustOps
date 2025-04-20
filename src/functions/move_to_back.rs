use ndarray::Axis;
use ndarray::Zip;
use std::fmt::Debug;

use super::abs::{self, abs_ndarray};
use super::expand::expand_at_dim;
use super::gather::gather;
use super::rearrange::rearrange_batch_mems_flag;
use super::slicing::slice_last_dim;
use super::sort::argsort_last_dim;
use super::sort::sort_last_dim;
use super::unsqueeze::{self, unsqueeze};
use super::wheres::where_op;
use super::wheres::where_value;
use ndarray::{Array, ArrayD, Dimension, IxDyn, RemoveAxis, ScalarOperand};

pub fn move_to_back<T>(x: &ArrayD<T>, value: T) -> ArrayD<T>
where
    T: num_traits::PrimInt + Clone + Debug + ScalarOperand,
{
    let equals = x.mapv(|elem| elem == value);
    let indices = argsort_last_dim(&equals);
    gather(&x, -1, &indices).unwrap()
}
