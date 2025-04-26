use RustOps::functions::pred_down;
use approx::assert_abs_diff_eq;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/pred_down_test.rs

#[test]
fn test_pred_down() {
    // Define file paths based on the Python script's saving pattern
    let up0_file = "data/pred_down_input_up0.npy";
    let up1_file = "data/pred_down_input_up1.npy";
    let up2_file = "data/pred_down_input_up2.npy";
    let layer0_file = "data/pred_down_layer0.npy";
    let layer1_file = "data/pred_down_layer1.npy";
    let layer2_file = "data/pred_down_layer2.npy"; // Note: Python saves 3 layers, but pred_down uses layers[0] and layers[1]
    let eps_file = "data/pred_down_eps.npy";
    let coeff_file = "data/pred_down_coeff.npy";
    let expected_down0_file = "data/pred_down_output_down0.npy"; // Corresponds to root_pred
    let expected_down1_file = "data/pred_down_output_down1.npy"; // Corresponds to h_sub_l_star for i=1
    let expected_down2_file = "data/pred_down_output_down2.npy"; // Corresponds to h_sub_l_star for i=0

    // Load input tensors
    let up0: ArrayD<f32> = read_npy(up0_file).expect("Failed to read up0 data");
    let up1: ArrayD<f32> = read_npy(up1_file).expect("Failed to read up1 data");
    let up2: ArrayD<f32> = read_npy(up2_file).expect("Failed to read up2 data");
    let layer0: ArrayD<f32> = read_npy(layer0_file).expect("Failed to read layer0 data");
    let layer1: ArrayD<f32> = read_npy(layer1_file).expect("Failed to read layer1 data");
    let layer2: ArrayD<f32> = read_npy(layer2_file).expect("Failed to read layer2 data");
    // layer2 is loaded but not directly used if pred_down expects layers.len() == upwards.len() - 1
    let _layer2: ArrayD<f32> = read_npy(layer2_file).expect("Failed to read layer2 data");

    // Load scalar inputs
    let eps_array: ArrayD<f32> = read_npy(eps_file).expect("Failed to read eps data");
    assert_eq!(eps_array.ndim(), 0, "Eps array should be 0-dimensional");
    let eps = eps_array
        .first()
        .copied()
        .expect("Eps array should contain a scalar");

    let coeff_array: ArrayD<f32> = read_npy(coeff_file).expect("Failed to read coeff data");
    assert_eq!(coeff_array.ndim(), 0, "Coeff array should be 0-dimensional");
    let coeff = coeff_array
        .first()
        .copied()
        .expect("Coeff array should contain a scalar");

    // Load expected output tensors
    let expected_down0: ArrayD<f32> =
        read_npy(expected_down0_file).expect("Failed to read expected down0 data");
    let expected_down1: ArrayD<f32> =
        read_npy(expected_down1_file).expect("Failed to read expected down1 data");
    let expected_down2: ArrayD<f32> =
        read_npy(expected_down2_file).expect("Failed to read expected down2 data");

    // Prepare inputs for the Rust function
    let upwards = [up0, up1, up2];
    // The layers connecting upwards[i] to upwards[i+1] are needed.
    // Python's net.layers[i] connects level i to i+1.
    // pred_down iterates i from upwards.len() - 2 down to 0.
    // It uses layers[i] to propagate from level i+1 down to level i.
    // So, layers[0] connects upwards[0] and upwards[1].
    // layers[1] connects upwards[1] and upwards[2].
    // The Rust function expects layers.len() == upwards.len() - 1
    let layers = [layer0, layer1, layer2]; // Use layers connecting the levels

    // Calculate using the Rust implementation
    let result_downwards = pred_down::pred_down(&upwards, &layers, eps, coeff);

    // Compare the results against the expected outputs
    assert_eq!(result_downwards.len(), 3, "Expected 3 downward tensors");
    // Note: The order in the returned `downwards` vector is [root_pred, pred_for_level_1, pred_for_level_0]
    // This matches the order of saving in the Python script: down0, down1, down2
    assert_abs_diff_eq!(result_downwards[0], expected_down0, epsilon = 1e-6);
    assert_abs_diff_eq!(result_downwards[1], expected_down1, epsilon = 1e-6);
    assert_abs_diff_eq!(result_downwards[2], expected_down2, epsilon = 1e-6);
}
