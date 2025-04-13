use RustOps::functions::argmaxi;
use approx::assert_abs_diff_eq;
use ndarray::Array3;
use ndarray_npy::read_npy;

#[test]
fn test_argmaxi_3d() {
    let xfile = "data/argmaxi_x.npy";
    let expected_file = "data/argmaxi_result_1e-08.npy";

    let x: Array3<f32> = read_npy(xfile).unwrap();
    let expected: Array3<f32> = read_npy(expected_file).unwrap();

    let result = argmaxi::argmaxi(&x, 1e-8);

    assert_abs_diff_eq!(result, expected, epsilon = 1e-8);
}
