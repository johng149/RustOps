use RustOps::functions::abs;
use ndarray::{Array2, Array3};
use ndarray_npy::read_npy;

#[test]
fn test_abs_ndarray_2d() {
    let xfile = "data/abs2d_abs_x.npy";
    let yfile = "data/abs2d_abs_y.npy";
    let x: Array2<f32> = read_npy(xfile).unwrap();
    let y: Array2<f32> = read_npy(yfile).unwrap();

    let result = abs::abs_ndarray(&x);

    assert_eq!(result, y);
}

#[test]
fn test_abs_ndarray_3d() {
    let xfile = "data/abs3d_abs_x.npy";
    let yfile = "data/abs3d_abs_y.npy";
    let x: Array3<f32> = read_npy(xfile).unwrap();
    let y: Array3<f32> = read_npy(yfile).unwrap();

    let result = abs::abs_ndarray(&x);

    assert_eq!(result, y);
}
