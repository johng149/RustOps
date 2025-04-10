use RustOps::functions::reduce;
use approx::assert_abs_diff_eq;
use ndarray::{Array, ArrayD, IxDyn};
use ndarray_npy::read_npy;

#[test]
fn test_reduce_bnm_nm() {
    // Test matrix multiplication as einsum: 'batch fields memories dim, batch fields dim -> batch fields memories'
    let x_file = "data/reduced_reduce_bnm_nm_reduce_sum_x.npy";
    let y_file = "data/reduced_reduce_bnm_nm_reduce_sum_y.npy";

    let x: ArrayD<f32> = read_npy(x_file).unwrap();
    let y: ArrayD<f32> = read_npy(y_file).unwrap();
    let equation = "bnm,bnm->nm";

    let result = reduce::reduce(&x, equation).unwrap();

    assert_abs_diff_eq!(result, y, epsilon = 1e-5);
}

#[test]
fn test_reduce_bfmd_bfm() {
    // Test reduction: 'batch fields memories dim -> batch fields memories'
    let x_file = "data/reduced_batch_fields_memories_reduce_bfmd_bfm_x.npy";
    let y_file = "data/reduced_batch_fields_memories_reduce_bfmd_bfm_y.npy";

    let x: ArrayD<f32> = read_npy(x_file).unwrap();
    let y: ArrayD<f32> = read_npy(y_file).unwrap();
    let equation = "bfmd,bfmd->bfm";

    let result = reduce::reduce(&x, equation).unwrap();

    assert_abs_diff_eq!(result, y, epsilon = 1e-5);
}

#[test]
fn test_reduce_bfd_bf() {
    // Test reduction: 'batch field dim -> batch field'
    let x_file = "data/reduced_batch_field_reduce_bfd_bf_x.npy";
    let y_file = "data/reduced_batch_field_reduce_bfd_bf_y.npy";

    let x: ArrayD<f32> = read_npy(x_file).unwrap();
    let y: ArrayD<f32> = read_npy(y_file).unwrap();
    let equation = "bfd,bfd->bf";

    let result = reduce::reduce(&x, equation).unwrap();

    assert_abs_diff_eq!(result, y, epsilon = 1e-5);
}

#[test]
fn test_reduce_bhccm_bh() {
    // Test reduction: 'batch hidden children c_mems -> batch hidden'
    let x_file = "data/reduced_batch_hidden_reduce_bhccm_bh_x.npy";
    let y_file = "data/reduced_batch_hidden_reduce_bhccm_bh_y.npy";

    let x: ArrayD<f32> = read_npy(x_file).unwrap();
    let y: ArrayD<f32> = read_npy(y_file).unwrap();
    let equation = "bhcm,bhcm->bh";

    let result = reduce::reduce(&x, equation).unwrap();

    assert_abs_diff_eq!(result, y, epsilon = 1e-5);
}
