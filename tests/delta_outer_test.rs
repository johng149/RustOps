use RustOps::functions::delta_outer;
use approx::assert_abs_diff_eq;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/delta_outer_test.rs

#[test]
fn test_delta_outer() {
    // Define file paths based on the Python script's saving pattern
    // Note: delta_outer uses downwards.last() and outer_mm.first()
    // In the python script, downwards has 3 elements, so last is downwards[2]
    // outer_mm corresponds to net.layers[0]
    let downwards2_file = "data/delta_outer_input_downwards2.npy";
    let layer0_file = "data/delta_outer_input_layer0.npy";
    let sensory_input_file = "data/delta_outer_input_sensory.npy";
    let expected_output_file = "data/delta_outer_output.npy";

    // Load input tensors
    let downwards2: ArrayD<f32> =
        read_npy(downwards2_file).expect("Failed to read downwards[2] data");
    let layer0: ArrayD<f32> = read_npy(layer0_file).expect("Failed to read layer0 data");
    let sensory_input: ArrayD<f32> =
        read_npy(sensory_input_file).expect("Failed to read sensory_input data");

    // Load expected output tensor
    let expected_output: ArrayD<f32> =
        read_npy(expected_output_file).expect("Failed to read expected output data");

    // Prepare inputs for the Rust function
    // delta_outer expects Vecs, even though it only uses specific elements internally
    // We only need the elements that will be accessed: downwards.last() and outer_mm.first()
    let downwards_vec = vec![downwards2]; // Only the last element is needed
    let outer_mm_vec = vec![layer0]; // Only the first element is needed

    // Call the Rust function
    let actual_output = delta_outer::delta_outer(&downwards_vec, &outer_mm_vec, &sensory_input);

    // Compare the actual output with the expected output
    assert_abs_diff_eq!(actual_output, expected_output, epsilon = 1e-5); // Adjust epsilon as needed
}
