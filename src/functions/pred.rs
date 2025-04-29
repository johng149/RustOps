use ndarray::{ArrayD, NdFloat};
use std::borrow::Cow;

use super::einsum::einsum_ndarray_dyn;
use super::einsum_specialized;
use super::reshape::reshape;
use super::unsqueeze::unsqueeze;

pub fn pred<'a, T>(parent_down_prop: &'a ArrayD<T>, parent_mem_matrix: &'a ArrayD<T>) -> ArrayD<T>
where
    T: NdFloat,
{
    let mm: Cow<'a, ArrayD<T>> = if parent_mem_matrix.ndim() == 4 {
        Cow::Borrowed(parent_mem_matrix)
    } else {
        Cow::Owned(unsqueeze(parent_mem_matrix, parent_mem_matrix.ndim() - 2))
    };
    let mm_shape = mm.shape();
    let (nodes, children_per_node, memories, dim) =
        (mm_shape[0], mm_shape[1], mm_shape[2], mm_shape[3]);
    let batch_size = parent_down_prop.shape()[0];
    let prediction =
        einsum_ndarray_dyn("ncmd,bnm->bncd", &[mm.as_ref(), parent_down_prop]).unwrap();
    // let prediction =
    //     einsum_specialized::einsum_ncmd_bnm_bncd_dyn(&mm.view(), &parent_down_prop.view()).unwrap();
    // usually, the specialized einsum is faster, but in this case, empirically, the generic einsum
    // is faster, I think it might have something to do with the shape of the arrays we happen to be
    // using, but need to investigate more
    let prediction = reshape(
        &prediction,
        &[
            batch_size as i64,
            (nodes * children_per_node) as i64,
            dim as i64,
        ],
    );
    prediction.unwrap()
}
