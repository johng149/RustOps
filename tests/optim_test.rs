use RustOps::functions::optim::optim;
use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use ndarray_npy::read_npy;
use num_traits::FromPrimitive; // Required for T::from(usize)

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/optim_test.rs

// Helper function to load a scalar value from a 0-dimensional npy tensor
fn load_scalar<T: ndarray_npy::ReadableElement + Copy>(path: &str) -> T {
    let arr: ArrayD<T> = read_npy(path).expect(&format!("Failed to read scalar from {}", path));
    *arr.first()
        .expect(&format!("Scalar tensor is empty in {}", path))
}

#[test]
fn test_optim() {
    // Define file paths based on the Python script's saving pattern
    let delta0_file = "data/optim_input_delta0.npy";
    let delta1_file = "data/optim_input_delta1.npy";
    let delta2_file = "data/optim_input_delta2.npy";

    let layer0_weights_initial_file = "data/optim_input_layer0_weights_initial.npy";
    let layer1_weights_initial_file = "data/optim_input_layer1_weights_initial.npy";
    let layer2_weights_initial_file = "data/optim_input_layer2_weights_initial.npy";

    let layer0_counts_file = "data/optim_input_layer0_counts.npy";
    let layer1_counts_file = "data/optim_input_layer1_counts.npy";
    let layer2_counts_file = "data/optim_input_layer2_counts.npy";

    let t_initial_file = "data/optim_input_t_initial.npy";
    let a_file = "data/optim_input_a.npy";

    let expected_layer0_weights_final_file = "data/optim_output_layer0_weights_final.npy";
    let expected_layer1_weights_final_file = "data/optim_output_layer1_weights_final.npy";
    let expected_layer2_weights_final_file = "data/optim_output_layer2_weights_final.npy";
    let expected_t_final_file = "data/optim_output_t_final.npy";

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

    // Load scalar inputs
    // Note: Python saves t as int32, load as i32 then cast to usize
    let t_initial_i32: i32 = load_scalar(t_initial_file);
    let t_initial: usize = t_initial_i32 as usize;
    let a: f32 = load_scalar(a_file);

    // Load expected output tensors
    let expected_layer0_weights_final: ArrayD<f32> = read_npy(expected_layer0_weights_final_file)
        .expect("Failed to read expected final layer0 weights");
    let expected_layer1_weights_final: ArrayD<f32> = read_npy(expected_layer1_weights_final_file)
        .expect("Failed to read expected final layer1 weights");
    let expected_layer2_weights_final: ArrayD<f32> = read_npy(expected_layer2_weights_final_file)
        .expect("Failed to read expected final layer2 weights");

    // Load expected scalar outputs
    let expected_t_final_i32: i32 = load_scalar(expected_t_final_file);
    let expected_t_final: usize = expected_t_final_i32 as usize;

    // Calculate expected growth threshold based on loaded final t and a
    let expected_t_final_float =
        f32::from_usize(expected_t_final).expect("Failed to convert final t to f32");
    let expected_growth_threshold = a / (expected_t_final_float + a);

    // Prepare inputs for the Rust function (order: outer to inner)
    let deltas_vec = vec![delta0, delta1, delta2];
    let layers_mm_vec = vec![
        layer0_weights_initial,
        layer1_weights_initial,
        layer2_weights_initial,
    ];
    let layer_counts_vec = vec![layer0_counts, layer1_counts, layer2_counts];

    // Call the Rust function
    let (actual_updated_layers_mm, actual_t_final, actual_growth_threshold) =
        optim(&deltas_vec, &layers_mm_vec, &layer_counts_vec, t_initial, a);

    // Compare the actual outputs with the expected outputs
    assert_eq!(
        actual_updated_layers_mm.len(),
        3,
        "Expected 3 updated layers"
    );
    assert_abs_diff_eq!(
        actual_updated_layers_mm[0],
        expected_layer0_weights_final,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        actual_updated_layers_mm[1],
        expected_layer1_weights_final,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        actual_updated_layers_mm[2],
        expected_layer2_weights_final,
        epsilon = 1e-6
    );

    assert_eq!(actual_t_final, expected_t_final, "Final t mismatch");
    assert_abs_diff_eq!(
        actual_growth_threshold,
        expected_growth_threshold,
        epsilon = 1e-9 // Use a smaller epsilon for scalar comparison if needed
    );
}
