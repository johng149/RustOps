use RustOps::functions::sqrt;
use approx::assert_abs_diff_eq;
use ndarray::{Array2, Array3};
use ndarray_npy::read_npy;

#[test]
fn test_sqrt_ndarray_2d() {
    let xfile = "data/sqrt2d_sqrt_x.npy";
    let yfile = "data/sqrt2d_sqrt_y.npy";
    let x: Array2<f32> = read_npy(xfile).unwrap();
    let y: Array2<f32> = read_npy(yfile).unwrap();

    let result = sqrt::sqrt_ndarray(&x);

    assert_abs_diff_eq!(result, y, epsilon = 1e-6);
}

#[test]
fn test_sqrt_ndarray_3d() {
    let xfile = "data/sqrt3d_sqrt_x.npy";
    let yfile = "data/sqrt3d_sqrt_y.npy";
    let x: Array3<f32> = read_npy(xfile).unwrap();
    let y: Array3<f32> = read_npy(yfile).unwrap();

    let result = sqrt::sqrt_ndarray(&x);

    assert_abs_diff_eq!(result, y, epsilon = 1e-6);
}
