use RustOps::functions::zeros_like;
use ndarray::{Array2, ArrayD};
use ndarray_npy::read_npy;

#[test]
fn test_zeros_like_ndarray_7d() {
    let xfile = "data/zeros_like_zeros_like_x.npy";
    let yfile = "data/zeros_like_zeros_like.npy";
    let x: ArrayD<f32> = read_npy(xfile).unwrap();
    let y: ArrayD<f32> = read_npy(yfile).unwrap();

    let result: ArrayD<f32> = zeros_like::zeros_like(&x);

    assert_eq!(result, y);
}
