use RustOps::functions::sort::sort_last_dim;
use approx::assert_abs_diff_eq;
use ndarray::array;
use ndarray::{Array2, Array3, ArrayD, Axis};
use ndarray_npy::read_npy;

#[test]
fn test_sort_big() {
    let xfile = "data/sort_sort_x.npy";
    let sortfile = "data/sort_sorted_v_dim6.npy";
    let indexfile = "data/sort_sorted_i_dim6.npy";

    let mut x: ArrayD<f32> = read_npy(xfile).unwrap();
    let sort: ArrayD<f32> = read_npy(sortfile).unwrap();
    let y: ArrayD<i64> = read_npy(indexfile).unwrap();

    let result = sort_last_dim(&mut x);

    assert_abs_diff_eq!(x, sort, epsilon = 1e-5);
}

#[test]
fn test_sort() {
    let xfile = "data/sortsmall_sort_x.npy";
    let sortfile = "data/sortsmall_sorted_v_dim2.npy";
    let indexfile = "data/sortsmall_sorted_i_dim2.npy";

    let mut x: ArrayD<f32> = read_npy(xfile).unwrap();
    let sort: ArrayD<f32> = read_npy(sortfile).unwrap();
    let y: ArrayD<i64> = read_npy(indexfile).unwrap();

    let result = sort_last_dim(&mut x);

    assert_abs_diff_eq!(x, sort, epsilon = 1e-5);
}
