use RustOps::functions::delta;
use approx::assert_abs_diff_eq;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

#[test]
fn test_delta() {
    // Define file paths based on the Python script's saving pattern
    let downwards0_file = "data/delta_input_downwards0.npy"; // Outer layer activation
    let downwards1_file = "data/delta_input_downwards1.npy"; // Middle layer activation
    let downwards2_file = "data/delta_input_downwards2.npy"; // Root layer activation

    let layer0_file = "data/delta_input_layer0.npy"; // Outer layer weights
    let layer1_file = "data/delta_input_layer1.npy"; // Middle layer weights
    let layer2_file = "data/delta_input_layer2.npy"; // Root layer weights

    let sensory_input_file = "data/delta_input_sensory.npy";

    let expected_delta0_file = "data/delta_output_delta0.npy"; // Outer delta
    let expected_delta1_file = "data/delta_output_delta1.npy"; // Middle delta
    let expected_delta2_file = "data/delta_output_delta2.npy"; // Root delta

    // Load input tensors
    let downwards0: ArrayD<f32> =
        read_npy(downwards0_file).expect("Failed to read downwards[0] data");
    let downwards1: ArrayD<f32> =
        read_npy(downwards1_file).expect("Failed to read downwards[1] data");
    let downwards2: ArrayD<f32> =
        read_npy(downwards2_file).expect("Failed to read downwards[2] data");

    let layer0: ArrayD<f32> = read_npy(layer0_file).expect("Failed to read layer0 data");
    let layer1: ArrayD<f32> = read_npy(layer1_file).expect("Failed to read layer1 data");
    let layer2: ArrayD<f32> = read_npy(layer2_file).expect("Failed to read layer2 data");

    let sensory_input: ArrayD<f32> =
        read_npy(sensory_input_file).expect("Failed to read sensory_input data");

    // Load expected output tensors
    let expected_delta0: ArrayD<f32> =
        read_npy(expected_delta0_file).expect("Failed to read expected delta0 data");
    let expected_delta1: ArrayD<f32> =
        read_npy(expected_delta1_file).expect("Failed to read expected delta1 data");
    let expected_delta2: ArrayD<f32> =
        read_npy(expected_delta2_file).expect("Failed to read expected delta2 data");

    // Prepare inputs for the Rust function (order: outer to inner)
    let downwards_vec = vec![downwards0, downwards1, downwards2];
    let layers_mm_vec = vec![layer0, layer1, layer2];

    // Call the Rust function
    let actual_deltas = delta::delta(&downwards_vec, &layers_mm_vec, sensory_input);

    // Compare the actual outputs with the expected outputs
    assert_eq!(
        actual_deltas.len(),
        3,
        "Expected 3 delta tensors, got {}",
        actual_deltas.len()
    );
    assert_abs_diff_eq!(actual_deltas[0], expected_delta0, epsilon = 1e-5); // Compare outer delta
    assert_abs_diff_eq!(actual_deltas[1], expected_delta1, epsilon = 1e-5); // Compare middle delta
    assert_abs_diff_eq!(actual_deltas[2], expected_delta2, epsilon = 1e-5); // Compare root delta
}
