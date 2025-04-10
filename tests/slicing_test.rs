use RustOps::functions::slicing;
use approx::assert_abs_diff_eq;
use ndarray::{Array2, Array3, ArrayD};
use ndarray_npy::read_npy;

#[test]
fn test_slicing_ccn1() {
    let xfile = "data/sliced_sliced[:, :, -1:]_x.npy";
    let yfile = "data/sliced_sliced[:, :, -1:]_y.npy";
    let x: ArrayD<f32> = read_npy(xfile).unwrap();
    let y: ArrayD<f32> = read_npy(yfile).unwrap();

    let result = slicing::slice_last_dim(&x).unwrap();
    assert_abs_diff_eq!(result, y, epsilon = 1e-5);
}

#[test]
fn test_slicing_cb() {
    let xfile = "data/sliced_sliced[:, :batch_size]_x.npy";
    let yfile = "data/sliced_3_sliced[:, :batch_size]_y.npy";
    let x: ArrayD<f32> = read_npy(xfile).unwrap();
    let y: ArrayD<f32> = read_npy(yfile).unwrap();

    let result = slicing::slice_second_dim(&x, 3).unwrap();
    assert_abs_diff_eq!(result, y, epsilon = 1e-5);
}
