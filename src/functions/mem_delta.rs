use ndarray::{Array, ArrayD, NdFloat};

use super::einsum::einsum_ndarray_dyn;
use super::pred::pred;
use super::reshape::reshape;
use super::unsqueeze::unsqueeze;

pub fn mem_delta<T>(
    parent_down_prop: &ArrayD<T>,
    parent_mem_matrix: &ArrayD<T>,
    child_down_prop: &ArrayD<T>,
) -> ArrayD<T>
where
    T: NdFloat,
{
    let shape: Vec<i64> = parent_mem_matrix
        .shape()
        .iter()
        .map(|&x| x as i64)
        .collect();
    let mm = if parent_mem_matrix.ndim() != 3 {
        parent_mem_matrix.clone() // Clone if we need to own it for unsqueeze
    } else {
        unsqueeze(parent_mem_matrix, parent_mem_matrix.ndim() - 2)
    };
    // Use mm_ref for subsequent operations that don't need ownership
    let mm_ref = if parent_mem_matrix.ndim() != 3 {
        parent_mem_matrix
    } else {
        &mm // Borrow the potentially unsqueezed array
    };

    let mm_shape = mm_ref.shape();
    let (nodes, children_per_node, memories, dim) =
        (mm_shape[0], mm_shape[1], mm_shape[2], mm_shape[3]);
    let batch_size = parent_down_prop.shape()[0];

    let prediction = pred(parent_down_prop, mm_ref);
    // Subtraction works with references
    let error = child_down_prop - &prediction;
    let error_reshaped = reshape(
        &error,
        &[
            batch_size as i64,
            nodes as i64,
            children_per_node as i64,
            dim as i64,
        ],
    )
    .unwrap();
    // einsum works with references
    let delta = einsum_ndarray_dyn("bncd,bnm->ncmd", &[&error_reshaped, parent_down_prop]).unwrap();
    reshape(&delta, &shape).unwrap()
}
