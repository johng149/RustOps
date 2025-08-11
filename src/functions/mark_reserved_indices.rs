use ndarray::Axis;
use ndarray::Zip;
use std::fmt::Debug;

use super::abs::{self, abs_ndarray};
use super::expand::expand_at_dim;
use super::rearrange::rearrange_batch_mems_flag;
use super::slicing::slice_last_dim;
use super::sort::argsort_last_dim;
use super::sort::sort_last_dim;
use super::unsqueeze::{self, unsqueeze};
use super::wheres::where_op;
use super::wheres::where_value;
use ndarray::{Array, ArrayD, Dimension, IxDyn, RemoveAxis, ScalarOperand};

pub fn mark_reserved_indices<T>(
    acts: &ArrayD<T>,
    usages: &ArrayD<T>,
    trigger_growth: &ArrayD<bool>,
    mark: i64, // should use negative values such as -2 as otherwise may be mistaken for normal usage value
) -> (ArrayD<i64>, ArrayD<i64>)
where
    T: num_traits::Float + Clone + Debug + ScalarOperand,
{
    // let used_values = sort_last_dim(acts);
    let used_indices = argsort_last_dim(acts);
    let reservations = slice_last_dim(&used_indices).unwrap();
    let reservations = where_value(trigger_growth, &reservations, -1).unwrap();
    let reservations = rearrange_batch_mems_flag(&reservations).unwrap();

    // let sorts = sort_last_dim(&usages);
    let sort_indices: ndarray::ArrayBase<
        ndarray::OwnedRepr<i64>,
        ndarray::Dim<ndarray::IxDynImpl>,
    > = argsort_last_dim(&usages);

    let reservation_mask = &reservations.mapv(|x| x != -1);

    let expanded_usages = unsqueeze(&sort_indices, 1);

    let expanded_usages: ndarray::ArrayBase<
        ndarray::OwnedRepr<i64>,
        ndarray::Dim<ndarray::IxDynImpl>,
    > = expand_at_dim(&expanded_usages, 1, reservations.shape()[1]).unwrap();

    let expanded_res = unsqueeze(&reservations, reservations.ndim());

    let expanded_res: ndarray::ArrayBase<
        ndarray::OwnedRepr<i64>,
        ndarray::Dim<ndarray::IxDynImpl>,
    > = expand_at_dim(&expanded_res, 2, usages.shape()[1]).unwrap();

    let unsqueezed_mask: ndarray::ArrayBase<
        ndarray::OwnedRepr<bool>,
        ndarray::Dim<ndarray::IxDynImpl>,
    > = unsqueeze(&reservation_mask, reservation_mask.ndim());

    let comparison = Zip::from(&expanded_res)
        .and(&expanded_usages)
        .map_collect(|res_val_ref, usage_val_ref| res_val_ref == usage_val_ref);

    let matches = Array::from_shape_fn(comparison.raw_dim(), |idx| {
        // Access the mask value at (i, j, 0) - broadcasting the last dimension
        let mask_idx = [idx[0], idx[1], 0];
        // Perform logical AND between comparison at full coordinate and mask with broadcast
        comparison[idx] && unsqueezed_mask[mask_idx]
    });

    let reduced_matches = matches.map_axis(Axis(1), |view| view.iter().any(|&x| x));

    let avail: ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<ndarray::IxDynImpl>> =
        where_value(&reduced_matches, &sort_indices, mark).unwrap();

    (avail, sort_indices)
}
