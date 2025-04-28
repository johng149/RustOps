use super::abs;
use super::argmaxi::argmaxi;
use super::expand::expand_at_dim;
use super::expand_for_batches::expand_for_batches;
use super::gather::gather;
use super::mark_reserved_indices::mark_reserved_indices;
use super::move_to_back::move_to_back;
use super::reduce::reduce;
use super::reduce_specialized;
use super::reshape::reshape;
use super::scatter::scatter;
use super::sum::sum_generic;
use super::transpose::transpose_dims;
use super::wheres::{where_op, where_value};
use super::zeros_like::zeros_like;
use super::{abs::abs_ndarray, expand};
use ndarray::{ArrayD, Axis, ScalarOperand};
use ndarray::{NdFloat, Zip};
use num_traits::{Float, NumCast, ToPrimitive, Zero};
use std::fmt::Debug;

/// Finds indices with maximum growth potential based on counts and values
///
/// # Arguments
/// * `x` - Values array of floating point type
/// * `counts` - Count values of integer type
/// * `eps` - Small epsilon value to avoid division by zero
/// * `threshold` - Threshold for considering growth potential
///
/// # Returns
/// * Indices of values with maximum growth potential
pub fn growth_argmaxi<T, U>(
    x: &ArrayD<T>,
    counts: &ArrayD<U>,
    eps: T,
    threshold: T,
    mark: i64,
) -> (ArrayD<T>, ArrayD<U>)
where
    T: Float + NumCast + ToPrimitive + Debug + Clone + ScalarOperand + NdFloat,
    U: Clone + ToPrimitive + Zero + Debug + num_traits::PrimInt + ScalarOperand,
{
    let x_shape = x.shape();
    let (batch_size, nodes, mems) = (x_shape[0], x_shape[1], x_shape[2]);
    let normal_path = argmaxi(x, eps);
    let condition = x.mapv(|value| if value > threshold { 1 } else { 0 });
    let condition_dim = condition.ndim();
    let trigger_growth = sum_generic(&condition, Axis(condition_dim - 1), true);
    let trigger_growth = trigger_growth.mapv(|value| value <= 0);
    let (avail, all) = mark_reserved_indices(&normal_path, &counts, &trigger_growth, mark);
    let avail = move_to_back(&avail, mark);
    let avail = expand_for_batches(&avail, batch_size).unwrap();
    let all = expand_for_batches(&all, batch_size).unwrap();

    let avail_is_mark = avail.mapv(|value| value == mark);
    let finalized = where_op(&avail_is_mark, &all, &avail).unwrap();
    let indices_sg = transpose_dims(&finalized, finalized.ndim() - 1, finalized.ndim() - 2);
    let reshape_shape: Vec<i64> = vec![batch_size as i64, -1, 1];
    let indices_sg = reshape(&indices_sg, &reshape_shape).unwrap();

    let mut growth_path = zeros_like(&normal_path);
    let values_of_interest = gather(&x, (x.ndim() - 1) as isize, &indices_sg).unwrap();
    let values_of_interest_eq_0 = values_of_interest.mapv(|value| value == T::zero());
    let values_of_interest = where_value(
        &values_of_interest_eq_0,
        &values_of_interest,
        NumCast::from(-1).unwrap(),
    )
    .unwrap();
    let scatter_dim = growth_path.ndim() - 1;
    let _ = scatter(
        &mut growth_path,
        scatter_dim as isize,
        &indices_sg,
        &values_of_interest,
    );
    let growth_path = abs_ndarray(&growth_path);
    let growth_path = argmaxi(&growth_path, eps);
    let trigger_growth = expand_at_dim(
        &trigger_growth,
        trigger_growth.ndim() - 1,
        normal_path.shape()[2],
    )
    .unwrap();
    let grown = where_op(&trigger_growth, &growth_path, &normal_path).unwrap();

    // let updated_counts = reduce(&grown, "bnm,bnm->nm").unwrap();
    let updated_counts = reduce_specialized::reduce_bnm_to_nm(&grown.view()).unwrap();
    let updated_counts_leq_0 = updated_counts.mapv(|value| value <= T::zero());
    let updated_counts = updated_counts.mapv(|value| NumCast::from(value).unwrap_or(U::zero()));
    let updated_counts = where_op(&updated_counts_leq_0, &counts, &updated_counts).unwrap();

    // return both grown and updated_counts
    (grown, updated_counts)
}
