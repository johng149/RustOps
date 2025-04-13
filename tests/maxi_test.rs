use RustOps::functions::maxi;
use ndarray::Array3;
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/maxi_test.rs

#[test]
fn test_maxi_3d() {
    let xfile = "data/maxi_x.npy";
    let expected_file = "data/maxi_scattered.npy";

    let x: Array3<f32> = read_npy(xfile).unwrap();
    let expected: Array3<f32> = read_npy(expected_file).unwrap();

    let result = maxi::maxi(&x);

    assert_eq!(result, expected);
}
