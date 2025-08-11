use RustOps::functions::root_down;
use approx::assert_abs_diff_eq;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

#[test]
fn test_root_down() {
    // Define file paths based on the Python script's saving pattern
    let up0_file = "data/root_down_up0.npy";
    let up1_file = "data/root_down_up1.npy";
    let root_up_file = "data/root_down_root_up.npy"; // This is upwards[-1]
    let initial_root_counts_file = "data/root_down_initial_root_counts.npy"; // This is layer_counts[-1]
    let eps_file = "data/root_down_eps.npy";
    let growth_threshold_file = "data/root_down_growth_threshold.npy";
    let expected_output_file = "data/root_down_output.npy"; // This is root_h_sub_l_star
    let expected_updated_counts_file = "data/root_down_updated_root_counts.npy";

    // Load input tensors
    let up0: ArrayD<f32> = read_npy(up0_file).expect("Failed to read up0 data");
    let up1: ArrayD<f32> = read_npy(up1_file).expect("Failed to read up1 data");
    let root_up: ArrayD<f32> = read_npy(root_up_file).expect("Failed to read root_up data");
    let initial_root_counts: ArrayD<f32> =
        read_npy(initial_root_counts_file).expect("Failed to read initial_root_counts data");

    // Load scalar inputs
    // read_npy loads scalars as 0-dimensional arrays
    let eps_array: ArrayD<f32> = read_npy(eps_file).expect("Failed to read eps data");
    assert_eq!(eps_array.ndim(), 0, "Eps array should be 0-dimensional");
    let eps = eps_array
        .first()
        .copied()
        .expect("Eps array should contain a scalar");

    let growth_threshold_array: ArrayD<f32> =
        read_npy(growth_threshold_file).expect("Failed to read growth_threshold data");
    assert_eq!(
        growth_threshold_array.ndim(),
        0,
        "Growth threshold array should be 0-dimensional"
    );
    let growth_threshold = growth_threshold_array
        .first()
        .copied()
        .expect("Growth threshold array should contain a scalar");

    // Load expected output tensors
    let expected_output: ArrayD<f32> =
        read_npy(expected_output_file).expect("Failed to read expected output data");
    let expected_updated_counts: ArrayD<f32> = read_npy(expected_updated_counts_file)
        .expect("Failed to read expected updated counts data");

    // Prepare inputs for the Rust function
    // The root_down function expects Vecs, but only uses the last element of each.
    // We provide the full upwards path as context, but only the relevant last count tensor.
    let upwards = vec![up0, up1, root_up];
    // IMPORTANT: root_down only uses the *last* element of layer_counts.
    // We load only the last count tensor saved by Python.
    let layer_counts = vec![initial_root_counts];
    let mark: i64 = -1; // Assuming a default mark value used implicitly or explicitly in Python's growth_argmaxi

    // Calculate using the Rust implementation
    let (result_output, result_updated_counts) =
        root_down::root_down(&upwards, &layer_counts, eps, growth_threshold, mark);

    // Compare the results against the expected outputs
    // Compare activation tensor (float)
    assert_abs_diff_eq!(result_output, expected_output, epsilon = 1e-5);

    // Compare counts tensor (integer)
    assert_eq!(result_updated_counts, expected_updated_counts);
}
