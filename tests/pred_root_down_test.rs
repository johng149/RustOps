use RustOps::functions::pred_root_down;
use approx::assert_abs_diff_eq;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/pred_root_down_test.rs

#[test]
fn test_pred_root_down() {
    // Define file paths based on the Python script's saving pattern
    let up0_file = "data/pred_root_down_input_up0.npy";
    let up1_file = "data/pred_root_down_input_up1.npy";
    let up2_file = "data/pred_root_down_input_up2.npy"; // This is the one used by pred_root_down
    let eps_file = "data/pred_root_down_eps.npy";
    let expected_output_file = "data/pred_root_down_output.npy";

    // Load input tensors
    let up0: ArrayD<f32> = read_npy(up0_file).expect("Failed to read up0 data");
    let up1: ArrayD<f32> = read_npy(up1_file).expect("Failed to read up1 data");
    let up2: ArrayD<f32> = read_npy(up2_file).expect("Failed to read up2 data");

    // Load scalar input eps
    let eps_array: ArrayD<f32> = read_npy(eps_file).expect("Failed to read eps data");
    assert_eq!(eps_array.ndim(), 0, "Eps array should be 0-dimensional");
    let eps = eps_array
        .first()
        .copied()
        .expect("Eps array should contain a scalar");

    // Load expected output tensor
    let expected_output: ArrayD<f32> =
        read_npy(expected_output_file).expect("Failed to read expected output data");

    // Prepare inputs for the Rust function
    // The pred_root_down function expects a Vec, but only uses the last element.
    let upwards = vec![up0, up1, up2];

    // Calculate using the Rust implementation
    let result_output = pred_root_down::pred_root_down(&upwards, eps);

    // Compare the result against the expected output
    assert_abs_diff_eq!(result_output, expected_output, epsilon = 1e-6); // Use appropriate epsilon
}
