use RustOps::functions::ones;
use ndarray::{Array2, Array3};
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/ones_test.rs

#[test]
fn test_ones_2d() {
    let yfile = "data/ones.npy";
    let y: Array2<f32> = read_npy(yfile).unwrap();

    // Create ones array with same shape as reference
    let result = ones::ones(y.dim());

    assert_eq!(result, y);
}
