use RustOps::functions::down_prop_parallel::down_prop_parallel;
use approx::assert_abs_diff_eq;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

#[test]
fn test_down_prop_parallel() {
    // Load input files
    let parent_h_file = "data/down_prop_parallel_parent_h_coeff0.5.npy";
    let parent_mm_file = "data/down_prop_parallel_parent_mm_coeff0.5.npy";
    let child_h_file = "data/down_prop_parallel_child_h_coeff0.5.npy";
    let expected_result_file = "data/down_prop_parallel_result_coeff0.5.npy";

    let parent_h: ArrayD<f32> = read_npy(parent_h_file).unwrap();
    let parent_mm: ArrayD<f32> = read_npy(parent_mm_file).unwrap();
    let child_h: ArrayD<f32> = read_npy(child_h_file).unwrap();
    let expected_result: ArrayD<f32> = read_npy(expected_result_file).unwrap();

    // Coefficient from the Python implementation
    let coef = 0.5;

    // Call the Rust implementation
    let result = down_prop_parallel(parent_h, parent_mm, child_h, coef);

    // Assert results match expected output
    assert_abs_diff_eq!(result, expected_result, epsilon = 1e-5);
}
