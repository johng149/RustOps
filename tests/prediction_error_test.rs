use RustOps::functions::prediction_error::prediction_error;
use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, IxDyn};
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/prediction_error_test.rs

fn run_prediction_error_test(name: &str) {
    // Define file paths based on the Python script's saving pattern
    let dir = "data";

    // --- Load Inputs ---
    let sensory_input_file = format!("{}/{}_input_sensory_input.npy", dir, name);
    let layer0_weights_file = format!("{}/{}_input_layer0_weights.npy", dir, name);
    let layer1_weights_file = format!("{}/{}_input_layer1_weights.npy", dir, name);
    let layer2_weights_file = format!("{}/{}_input_layer2_weights.npy", dir, name);
    let eps_file = format!("{}/{}_input_eps.npy", dir, name);
    let coeff_file = format!("{}/{}_input_coeff.npy", dir, name);
    let rho_file = format!("{}/{}_input_rho.npy", dir, name);

    let sensory_input: ArrayD<f32> =
        read_npy(&sensory_input_file).expect("Failed to read sensory_input");
    let layer0_weights: ArrayD<f32> =
        read_npy(&layer0_weights_file).expect("Failed to read layer0_weights");
    let layer1_weights: ArrayD<f32> =
        read_npy(&layer1_weights_file).expect("Failed to read layer1_weights");
    let layer2_weights: ArrayD<f32> =
        read_npy(&layer2_weights_file).expect("Failed to read layer2_weights");

    let eps_array: ArrayD<f32> = read_npy(&eps_file).expect("Failed to read eps");
    assert_eq!(eps_array.ndim(), 0, "eps array should be 0-dimensional");
    let eps = eps_array
        .first()
        .copied()
        .expect("eps array should contain a scalar");

    let coeff_array: ArrayD<f32> = read_npy(&coeff_file).expect("Failed to read coeff");
    assert_eq!(coeff_array.ndim(), 0, "coeff array should be 0-dimensional");
    let coeff = coeff_array
        .first()
        .copied()
        .expect("coeff array should contain a scalar");

    let rho_array: ArrayD<f32> = read_npy(&rho_file).expect("Failed to read rho");
    assert_eq!(rho_array.ndim(), 0, "rho array should be 0-dimensional");
    let rho = rho_array
        .first()
        .copied()
        .expect("rho array should contain a scalar");

    // --- Load Expected Output ---
    let expected_error_file = format!("{}/{}_output_error.npy", dir, name);
    let expected_error_array: ArrayD<f32> =
        read_npy(&expected_error_file).expect("Failed to read expected_error");
    assert_eq!(
        expected_error_array.ndim(),
        0,
        "expected_error array should be 0-dimensional"
    );
    let expected_error = expected_error_array
        .first()
        .copied()
        .expect("expected_error array should contain a scalar");

    // --- Prepare Inputs for Rust Function ---
    let layers = vec![layer0_weights, layer1_weights, layer2_weights];

    // --- Call the Rust Function ---
    let actual_error = prediction_error(&layers, sensory_input, rho, eps, coeff);

    // --- Compare Results ---
    // Use approx for floating-point comparisons
    assert_abs_diff_eq!(
        actual_error,
        expected_error,
        epsilon = 1e-5 // Adjust tolerance as needed
    );
}

#[test]
fn test_prediction_error_untrained() {
    run_prediction_error_test("prediction_error_untrained");
}

#[test]
fn test_prediction_error_trained() {
    run_prediction_error_test("prediction_error_trained");
}
