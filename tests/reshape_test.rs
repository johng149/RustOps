use RustOps::functions::reshape;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/reshape_test.rs

const INPUT_PREFIX: &str = "reshape_reshape_original";
const OUTPUT_PREFIX: &str = "reshape_reshape_reshaped";
const NPY_SUFFIX: &str = ".npy";

/// Parses the filename to extract reshape parameters.
/// Expected format: "reshape_reshape_reshaped.npy"
fn parse_test_filename(filename: &str) -> Option<String> {
    if !filename.starts_with(OUTPUT_PREFIX) || !filename.ends_with(NPY_SUFFIX) {
        return None;
    }

    // Return the base name that can be used to find the corresponding input file
    let base_name = filename
        .strip_prefix(OUTPUT_PREFIX)?
        .strip_suffix(NPY_SUFFIX)?;

    Some(format!("{}{}{}", INPUT_PREFIX, base_name, NPY_SUFFIX))
}

/// Given directory path, discover all reshape test files and return their paths.
fn discover_reshape_test_files(dir: &str) -> Vec<String> {
    let mut test_files = Vec::new();
    let entries = std::fs::read_dir(dir).unwrap();

    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_file() {
            if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                if filename.starts_with(OUTPUT_PREFIX) && filename.ends_with(NPY_SUFFIX) {
                    test_files.push(filename.to_string());
                }
            }
        }
    }

    test_files
}

#[test]
fn test_reshape_ndarray() {
    let test_files = discover_reshape_test_files("data");

    for output_file in test_files {
        if let Some(input_file) = parse_test_filename(&output_file) {
            println!(
                "Running reshape test: input = {}, output = {}",
                input_file, output_file
            );

            let input_path = format!("data/{}", input_file);
            let output_path = format!("data/{}", output_file);

            let input: ArrayD<f32> = read_npy(&input_path).unwrap();
            let expected_output: ArrayD<f32> = read_npy(&output_path).unwrap();

            // Infer the new shape from the expected output
            let new_shape = expected_output.shape().to_vec();

            // Convert Vec<usize> to Vec<i64> for reshape function
            let new_shape_i64: Vec<i64> = new_shape.iter().map(|&x| x as i64).collect();

            // Call the reshape function (implementation will depend on your API)
            let result = reshape::reshape(&input, &new_shape_i64).unwrap();

            assert_eq!(result.shape(), expected_output.shape());
            assert_eq!(result, expected_output);
        } else {
            panic!("Invalid test file format: {}", output_file);
        }
    }
}
