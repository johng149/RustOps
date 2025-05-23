use RustOps::functions::chunking;
use approx::assert_abs_diff_eq;
use ndarray::ArrayD;
use ndarray_npy::read_npy;
use regex::Regex;

// Helper function to parse parameters (chunk_height, chunk_width, pad_value)
// from the expected output filename.
// Example filename format: "data/some_basename_result_ch<CH>_cw<CW>_pad<PAD>.npy"
fn parse_params_from_filename(filename: &str) -> (usize, usize, f32) {
    let re = Regex::new(r"_result_ch(\d+)_cw(\d+)_pad(-?\d+)\.npy$")
        .expect("Failed to compile regex for parsing filename parameters.");

    if let Some(caps) = re.captures(filename) {
        let chunk_height = caps
            .get(1)
            .unwrap()
            .as_str()
            .parse::<usize>()
            .expect("Failed to parse chunk_height from filename.");
        let chunk_width = caps
            .get(2)
            .unwrap()
            .as_str()
            .parse::<usize>()
            .expect("Failed to parse chunk_width from filename.");
        let pad_value = caps
            .get(3)
            .unwrap()
            .as_str()
            .parse::<f32>()
            .expect("Failed to parse pad_value from filename.");
        (chunk_height, chunk_width, pad_value)
    } else {
        panic!(
            "Could not parse parameters (ch, cw, pad) from filename: {}",
            filename
        );
    }
}

// Helper function to run a single chunking test case.
// input_file_basename: The base name used in the Python script's `create_chunking` (e.g., "chunking_32x32_ch8cw8_float").
// output_file_param_suffix: The suffix part of the output filename that contains parameters (e.g., "_result_ch8_cw8_pad0").
fn run_chunking_test(input_file_basename: &str, output_file_param_suffix: &str) {
    let input_file = format!("data/{}_x.npy", input_file_basename);
    let expected_output_file = format!(
        "data/{}{}.npy",
        input_file_basename, output_file_param_suffix
    );

    // Parse parameters from the expected output filename
    let (chunk_height, chunk_width, pad_value) = parse_params_from_filename(&expected_output_file);

    // Load input tensor (assuming f32 based on Python script's common usage)
    let input_tensor: ArrayD<f32> = read_npy(&input_file)
        .unwrap_or_else(|e| panic!("Failed to read input tensor '{}': {:?}", input_file, e));

    // Load expected output tensor
    let expected_output: ArrayD<f32> = read_npy(&expected_output_file).unwrap_or_else(|e| {
        panic!(
            "Failed to read expected output tensor '{}': {:?}",
            expected_output_file, e
        )
    });

    // Call the Rust chunking function
    let actual_output =
        chunking::chunking(&input_tensor, chunk_height, chunk_width, pad_value).unwrap_or_else(
            |e| {
                panic!(
                    "Chunking function failed for base name '{}' with params (ch: {}, cw: {}, pad: {}): {:?}",
                    input_file_basename, chunk_height, chunk_width, pad_value, e
                )
            },
        );

    // Compare the actual output with the expected output
    assert_abs_diff_eq!(actual_output, expected_output, epsilon = 1e-5);
}

#[test]
fn test_chunking_32x32_ch8cw8_float_pad0() {
    run_chunking_test("chunking_32x32_ch8cw8_float", "_result_ch8_cw8_pad0");
}

#[test]
fn test_chunking_5x7_ch2cw3_pad0() {
    // Python name: "chunking_5x7_ch2cw3_int_pad0", dtype=torch.float32
    run_chunking_test("chunking_5x7_ch2cw3_int_pad0", "_result_ch2_cw3_pad0");
}

#[test]
fn test_chunking_3x3_ch2cw2_pad5() {
    // Python name: "chunking_3x3_ch2cw2_float16_pad5", dtype=torch.float32
    run_chunking_test("chunking_3x3_ch2cw2_float16_pad5", "_result_ch2_cw2_pad5");
}

#[test]
fn test_chunking_7x7_ch7cw7_full_pad0() {
    run_chunking_test("chunking_7x7_ch7cw7_full", "_result_ch7_cw7_pad0");
}

#[test]
fn test_chunking_2x2_ch1cw1_pad0() {
    // Python name: "chunking_2x2_ch1cw1_int8", dtype=torch.float32
    run_chunking_test("chunking_2x2_ch1cw1_int8", "_result_ch1_cw1_pad0");
}
