use ndarray::{ArrayD, NdFloat};

use super::einsum::einsum_ndarray_dyn;
use super::reshape::reshape;
use super::unsqueeze::unsqueeze;

pub fn pred<T>(parent_down_prop: &ArrayD<T>, parent_mem_matrix: ArrayD<T>) -> ArrayD<T>
where
    T: NdFloat,
{
    let mm = if parent_mem_matrix.ndim() == 4 {
        parent_mem_matrix
    } else {
        unsqueeze(&parent_mem_matrix, parent_mem_matrix.ndim() - 2)
    };
    let mm_shape = mm.shape();
    let (nodes, children_per_node, memories, dim) =
        (mm_shape[0], mm_shape[1], mm_shape[2], mm_shape[3]);
    let batch_size = parent_down_prop.shape()[0];
    let prediction = einsum_ndarray_dyn("ncmd,bnm->bncd", &[&mm, parent_down_prop]).unwrap();
    let proposed_shape = [
        batch_size as i64,
        (nodes * children_per_node) as i64,
        dim as i64,
    ];
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
