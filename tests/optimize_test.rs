use RustOps::functions::optimize::optimize;
use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, IxDyn};
use ndarray_npy::read_npy;
use num_traits::Zero; // Import Zero trait for initial growth_threshold

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/optimize_test.rs

#[test]
fn test_optimize_3layer() {
    // Define file paths based on the Python script's saving pattern
    let name = "optimize_3layer";
    let dir = "data";

    // --- Load Inputs ---
    let sensory_input_file = format!("{}/{}_input_sensory_input.npy", dir, name);
    let initial_layer0_file = format!("{}/{}_input_layer0_weights_initial.npy", dir, name);
    let initial_layer1_file = format!("{}/{}_input_layer1_weights_initial.npy", dir, name);
    let initial_layer2_file = format!("{}/{}_input_layer2_weights_initial.npy", dir, name);
    let initial_count0_file = format!("{}/{}_input_layer0_counts_initial.npy", dir, name);
    let initial_count1_file = format!("{}/{}_input_layer1_counts_initial.npy", dir, name);
    let initial_count2_file = format!("{}/{}_input_layer2_counts_initial.npy", dir, name);
    let initial_t_file = format!("{}/{}_input_t_initial.npy", dir, name);
    let eps_file = format!("{}/{}_input_eps.npy", dir, name);
    let coeff_file = format!("{}/{}_input_coeff.npy", dir, name);
    let alpha_file = format!("{}/{}_input_alpha.npy", dir, name);
    let rho_file = format!("{}/{}_input_rho.npy", dir, name);

    let sensory_input: ArrayD<f32> =
        read_npy(&sensory_input_file).expect("Failed to read sensory_input");
    let initial_layer0: ArrayD<f32> =
        read_npy(&initial_layer0_file).expect("Failed to read initial_layer0");
    let initial_layer1: ArrayD<f32> =
        read_npy(&initial_layer1_file).expect("Failed to read initial_layer1");
    let initial_layer2: ArrayD<f32> =
        read_npy(&initial_layer2_file).expect("Failed to read initial_layer2");
    let initial_count0: ArrayD<f32> =
        read_npy(&initial_count0_file).expect("Failed to read initial_count0"); // Assuming counts are i64
    let initial_count1: ArrayD<f32> =
        read_npy(&initial_count1_file).expect("Failed to read initial_count1");
    let initial_count2: ArrayD<f32> =
        read_npy(&initial_count2_file).expect("Failed to read initial_count2");

    let initial_t_array: ArrayD<i32> = read_npy(&initial_t_file).expect("Failed to read initial_t");
    assert_eq!(initial_t_array.ndim(), 0, "t array should be 0-dimensional");
    let initial_t = initial_t_array
        .first()
        .copied()
        .expect("t array should contain a scalar") as usize;

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

    let alpha_array: ArrayD<f32> = read_npy(&alpha_file).expect("Failed to read alpha");
    assert_eq!(alpha_array.ndim(), 0, "alpha array should be 0-dimensional");
    let alpha = alpha_array // Parameter 'a' in optimize function
        .first()
        .copied()
        .expect("alpha array should contain a scalar");

    let rho_array: ArrayD<f32> = read_npy(&rho_file).expect("Failed to read rho");
    assert_eq!(rho_array.ndim(), 0, "rho array should be 0-dimensional");
    let rho = rho_array
        .first()
        .copied()
        .expect("rho array should contain a scalar");

    // --- Load Expected Outputs ---
    let final_layer0_file = format!("{}/{}_output_layer0_weights_final.npy", dir, name);
    let final_layer1_file = format!("{}/{}_output_layer1_weights_final.npy", dir, name);
    let final_layer2_file = format!("{}/{}_output_layer2_weights_final.npy", dir, name);
    let final_count0_file = format!("{}/{}_output_layer0_counts_final.npy", dir, name);
    let final_count1_file = format!("{}/{}_output_layer1_counts_final.npy", dir, name);
    let final_count2_file = format!("{}/{}_output_layer2_counts_final.npy", dir, name);
    let final_t_file = format!("{}/{}_output_t_final.npy", dir, name);

    let expected_layer0: ArrayD<f32> =
        read_npy(&final_layer0_file).expect("Failed to read expected_layer0");
    let expected_layer1: ArrayD<f32> =
        read_npy(&final_layer1_file).expect("Failed to read expected_layer1");
    let expected_layer2: ArrayD<f32> =
        read_npy(&final_layer2_file).expect("Failed to read expected_layer2");
    let expected_count0: ArrayD<f32> =
        read_npy(&final_count0_file).expect("Failed to read expected_count0");
    let expected_count1: ArrayD<f32> =
        read_npy(&final_count1_file).expect("Failed to read expected_count1");
    let expected_count2: ArrayD<f32> =
        read_npy(&final_count2_file).expect("Failed to read expected_count2");

    let expected_t_array: ArrayD<i32> = read_npy(&final_t_file).expect("Failed to read expected_t");
    assert_eq!(
        expected_t_array.ndim(),
        0,
        "t array should be 0-dimensional"
    );
    let expected_t = expected_t_array
        .first()
        .copied()
        .expect("t array should contain a scalar") as usize;

    // --- Prepare Inputs for Rust Function ---
    let layers = [
        initial_layer0.into_dyn(), // Use into_dyn() if needed, or ensure consistent IxDyn usage
        initial_layer1.into_dyn(),
        initial_layer2.into_dyn(),
    ];

    let initial_count0_i64 = initial_count0.mapv(|x| x as i64);
    let initial_count1_i64 = initial_count1.mapv(|x| x as i64);
    let initial_count2_i64 = initial_count2.mapv(|x| x as i64);

    let layer_counts = vec![
        initial_count0_i64.into_dyn(),
        initial_count1_i64.into_dyn(),
        initial_count2_i64.into_dyn(),
    ];

    // Values not saved by Python script, using defaults from SparseHopfield
    let init_growth_threshold_file = format!("{}/{}_input_growth_threshold.npy", dir, name);
    let initial_growth_threshold: ArrayD<f32> =
        read_npy(init_growth_threshold_file).expect("Failed to read initial_growth_threshold");
    assert_eq!(
        initial_growth_threshold.ndim(),
        0,
        "initial_growth_threshold array should be 0-dimensional"
    );
    let initial_growth_threshold = initial_growth_threshold
        .first()
        .copied()
        .expect("initial_growth_threshold array should contain a scalar");
    let final_threshold_file = format!("{}/{}_output_growth_threshold_final.npy", dir, name);
    let final_growth_threshold: ArrayD<f32> =
        read_npy(final_threshold_file).expect("Failed to read final_growth_threshold");
    assert_eq!(
        final_growth_threshold.ndim(),
        0,
        "final_growth_threshold array should be 0-dimensional"
    );
    let final_growth_threshold = final_growth_threshold
        .first()
        .copied()
        .expect("final_growth_threshold array should contain a scalar");
    let mark: i64 = -2; // Default mark value

    // --- Execute the Rust optimize function ---
    let (result_layers, result_counts, result_t, _result_growth_threshold) = optimize::<f32, i64>(
        &layers,
        &layer_counts,
        sensory_input.into_dyn(),
        initial_t,
        alpha, // 'a' parameter in optimize
        rho,
        eps,
        coeff,
        initial_growth_threshold,
        mark,
    );

    // --- Compare Results ---
    assert_eq!(result_layers.len(), 3, "Expected 3 result layers");
    assert_eq!(result_counts.len(), 3, "Expected 3 result counts");

    // Compare layers (floating point)
    assert_abs_diff_eq!(
        result_layers[0].view(),
        expected_layer0.view(),
        epsilon = 1e-5
    );
    assert_abs_diff_eq!(
        result_layers[1].view(),
        expected_layer1.view(),
        epsilon = 1e-5
    );
    assert_abs_diff_eq!(
        result_layers[2].view(),
        expected_layer2.view(),
        epsilon = 1e-5
    );

    // Compare counts (integer)
    let result_counts0 = result_counts[0].mapv(|x| x as f32);
    let result_counts1 = result_counts[1].mapv(|x| x as f32);
    let result_counts2 = result_counts[2].mapv(|x| x as f32);
    assert_abs_diff_eq!(result_counts0, expected_count0, epsilon = 1e-5);
    assert_abs_diff_eq!(result_counts1, expected_count1, epsilon = 1e-5);
    assert_abs_diff_eq!(result_counts2, expected_count2, epsilon = 1e-5);

    // Compare t (integer)
    assert_eq!(result_t, expected_t);
    // Compare growth threshold (floating point)
    assert_abs_diff_eq!(
        _result_growth_threshold,
        final_growth_threshold,
        epsilon = 1e-5
    );
}
