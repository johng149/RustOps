use RustOps::functions::optimize::optimize;
use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, IxDyn, s};
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/incremental_optimize_test.rs

#[test]
fn test_incremental_optimize_3layer() {
    // Define file paths based on the Python script's saving pattern
    let name = "incremental_optimize_3layer";
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
    let init_growth_threshold_file = format!("{}/{}_input_growth_threshold.npy", dir, name);

    let sensory_input: ArrayD<f32> =
        read_npy(&sensory_input_file).expect("Failed to read sensory_input");
    let initial_layer0: ArrayD<f32> =
        read_npy(&initial_layer0_file).expect("Failed to read initial_layer0");
    let initial_layer1: ArrayD<f32> =
        read_npy(&initial_layer1_file).expect("Failed to read initial_layer1");
    let initial_layer2: ArrayD<f32> =
        read_npy(&initial_layer2_file).expect("Failed to read initial_layer2");
    let initial_count0: ArrayD<f32> =
        read_npy(&initial_count0_file).expect("Failed to read initial_count0");
    let initial_count1: ArrayD<f32> =
        read_npy(&initial_count1_file).expect("Failed to read initial_count1");
    let initial_count2: ArrayD<f32> =
        read_npy(&initial_count2_file).expect("Failed to read initial_count2");

    let initial_t_array: ArrayD<i32> = read_npy(&initial_t_file).expect("Failed to read initial_t");
    let initial_t = initial_t_array.first().copied().unwrap() as usize;

    let eps_array: ArrayD<f32> = read_npy(&eps_file).expect("Failed to read eps");
    let eps = eps_array.first().copied().unwrap();

    let coeff_array: ArrayD<f32> = read_npy(&coeff_file).expect("Failed to read coeff");
    let coeff = coeff_array.first().copied().unwrap();

    let alpha_array: ArrayD<f32> = read_npy(&alpha_file).expect("Failed to read alpha");
    let alpha = alpha_array.first().copied().unwrap();

    let rho_array: ArrayD<f32> = read_npy(&rho_file).expect("Failed to read rho");
    let rho = rho_array.first().copied().unwrap();

    let initial_growth_threshold_array: ArrayD<f32> =
        read_npy(init_growth_threshold_file).expect("Failed to read initial_growth_threshold");
    let initial_growth_threshold = initial_growth_threshold_array.first().copied().unwrap();

    // --- Load Expected Outputs ---
    let final_layer0_file = format!("{}/{}_output_layer0_weights_final.npy", dir, name);
    let final_layer1_file = format!("{}/{}_output_layer1_weights_final.npy", dir, name);
    let final_layer2_file = format!("{}/{}_output_layer2_weights_final.npy", dir, name);
    let final_count0_file = format!("{}/{}_output_layer0_counts_final.npy", dir, name);
    let final_count1_file = format!("{}/{}_output_layer1_counts_final.npy", dir, name);
    let final_count2_file = format!("{}/{}_output_layer2_counts_final.npy", dir, name);
    let final_t_file = format!("{}/{}_output_t_final.npy", dir, name);
    let final_threshold_file = format!("{}/{}_output_growth_threshold_final.npy", dir, name);

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
    let expected_t = expected_t_array.first().copied().unwrap() as usize;
    let final_growth_threshold_array: ArrayD<f32> =
        read_npy(final_threshold_file).expect("Failed to read final_growth_threshold");
    let expected_growth_threshold = final_growth_threshold_array.first().copied().unwrap();

    // --- Prepare for Incremental Optimization ---
    let mark: i64 = -2; // Default mark value from Python implementation

    // --- First Optimization Step (on first sample) ---
    let layers_step1 = [
        initial_layer0.into_dyn(),
        initial_layer1.into_dyn(),
        initial_layer2.into_dyn(),
    ];
    let counts_step1 = vec![
        initial_count0.into_dyn(),
        initial_count1.into_dyn(),
        initial_count2.into_dyn(),
    ];
    let input_step1 = sensory_input.slice(s![0..1, .., ..]).to_owned();

    let (layers_step2, counts_step2, t_step2, growth_threshold_step2) = optimize::<f32>(
        &layers_step1,
        &counts_step1,
        input_step1.into_dyn(),
        initial_t,
        alpha,
        rho,
        eps,
        coeff,
        initial_growth_threshold,
        mark,
    );

    print!("Second optimization step");

    // --- Second Optimization Step (on remaining samples) ---
    let input_step2 = sensory_input.slice(s![1.., .., ..]).to_owned();

    let (result_layers, result_counts, result_t, result_growth_threshold) = optimize::<f32>(
        &layers_step2,
        &counts_step2,
        input_step2.into_dyn(),
        t_step2,
        alpha,
        rho,
        eps,
        coeff,
        growth_threshold_step2,
        mark,
    );

    // --- Compare Final Results ---
    assert_eq!(result_layers.len(), 3, "Expected 3 result layers");
    assert_eq!(result_counts.len(), 3, "Expected 3 result counts");

    // Compare layers
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

    // Compare counts
    let result_counts0 = result_counts[0].mapv(|x| x as f32);
    let result_counts1 = result_counts[1].mapv(|x| x as f32);
    let result_counts2 = result_counts[2].mapv(|x| x as f32);
    assert_abs_diff_eq!(result_counts0, expected_count0, epsilon = 1e-5);
    assert_abs_diff_eq!(result_counts1, expected_count1, epsilon = 1e-5);
    assert_abs_diff_eq!(result_counts2, expected_count2, epsilon = 1e-5);

    // Compare t
    assert_eq!(result_t, expected_t);

    // Compare growth threshold
    assert_abs_diff_eq!(
        result_growth_threshold,
        expected_growth_threshold,
        epsilon = 1e-5
    );
}
