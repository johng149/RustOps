use RustOps::functions::optim_outer;
use approx::assert_abs_diff_eq;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/optim_outer_test.rs

#[test]
fn test_optim_outer() {
    // Define file paths based on the Python script's saving pattern
    let delta0_file = "data/optim_outer_input_delta0.npy";
    let delta1_file = "data/optim_outer_input_delta1.npy";
    let delta2_file = "data/optim_outer_input_delta2.npy";

    let layer0_weights_initial_file = "data/optim_outer_input_layer0_weights_initial.npy";
    let layer1_weights_initial_file = "data/optim_outer_input_layer1_weights_initial.npy";
    let layer2_weights_initial_file = "data/optim_outer_input_layer2_weights_initial.npy";

    let layer0_counts_file = "data/optim_outer_input_layer0_counts.npy";
    let layer1_counts_file = "data/optim_outer_input_layer1_counts.npy";
    let layer2_counts_file = "data/optim_outer_input_layer2_counts.npy";

    let expected_layer0_weights_final_file = "data/optim_outer_output_layer0_weights_final.npy";

    // Load input tensors
    let delta0: ArrayD<f32> = read_npy(delta0_file).expect("Failed to read delta0 data");
    let delta1: ArrayD<f32> = read_npy(delta1_file).expect("Failed to read delta1 data");
    let delta2: ArrayD<f32> = read_npy(delta2_file).expect("Failed to read delta2 data");

    let layer0_weights_initial: ArrayD<f32> =
        read_npy(layer0_weights_initial_file).expect("Failed to read initial layer0 weights");
    let layer1_weights_initial: ArrayD<f32> =
        read_npy(layer1_weights_initial_file).expect("Failed to read initial layer1 weights");
    let layer2_weights_initial: ArrayD<f32> =
        read_npy(layer2_weights_initial_file).expect("Failed to read initial layer2 weights");

    let layer0_counts: ArrayD<f32> =
        read_npy(layer0_counts_file).expect("Failed to read layer0 counts");
    let layer1_counts: ArrayD<f32> =
        read_npy(layer1_counts_file).expect("Failed to read layer1 counts");
    let layer2_counts: ArrayD<f32> =
        read_npy(layer2_counts_file).expect("Failed to read layer2 counts");

    // Load expected output tensor
    let expected_layer0_weights_final: ArrayD<f32> = read_npy(expected_layer0_weights_final_file)
        .expect("Failed to read expected final layer0 weights");

    // Prepare inputs for the Rust function (order: outer to inner)
    let deltas_vec = vec![delta0, delta1, delta2];
    let layers_mm_vec = vec![
        layer0_weights_initial,
        layer1_weights_initial,
        layer2_weights_initial,
    ];
    let layer_counts_vec = vec![layer0_counts, layer1_counts, layer2_counts];

    // Call the Rust function
    let actual_layer0_weights_final =
        optim_outer::optim_outer(&deltas_vec, &layers_mm_vec, &layer_counts_vec);

    // Compare the actual output with the expected output
    // Note: The Python script also saves final layer1 and layer2 weights,
    // but optim_outer only modifies and returns layer0 weights.
    assert_abs_diff_eq!(
        actual_layer0_weights_final,
        expected_layer0_weights_final,
        epsilon = 1e-6 // Adjust epsilon based on expected precision
    );
}
