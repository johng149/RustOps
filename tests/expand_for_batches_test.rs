use RustOps::functions::expand_for_batches::expand_for_batches;
use ndarray::ArrayD;
use ndarray_npy::read_npy;
use std::path::Path;

const INPUT_PREFIX: &str = "expand_for_batches_";
const RESULT_SUFFIX: &str = "_result.npy";
const INPUT_SUFFIX: &str = "_x.npy";

/// Parses the filename to extract batch size
/// Expected format: "expand_for_batches_<batch_size>_x.npy"
fn parse_test_filename(filename: &str) -> Option<usize> {
    if !filename.starts_with(INPUT_PREFIX) || !filename.ends_with(INPUT_SUFFIX) {
        return None;
    }

    // Extract the middle part between prefix and suffix
    let batch_size_part = filename
        .strip_prefix(INPUT_PREFIX)?
        .strip_suffix(INPUT_SUFFIX)?;

    // Parse the batch size
    batch_size_part.parse::<usize>().ok()
}

/// Discover all expand_for_batches test files and return their paths
fn discover_test_files(dir: &str) -> Vec<String> {
    let mut test_files = Vec::new();
    let entries = std::fs::read_dir(dir).unwrap();

    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_file() {
            if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                if filename.starts_with(INPUT_PREFIX) && filename.ends_with(INPUT_SUFFIX) {
                    test_files.push(filename.to_string());
                }
            }
        }
    }

    test_files
}

#[test]
fn test_expand_for_batches() {
    let test_files = discover_test_files("data");
    assert!(!test_files.is_empty(), "No test files found");

    for test_file in test_files {
        if let Some(batch_size) = parse_test_filename(&test_file) {
            println!("Running test for batch_size: {}", batch_size);

            // Load input tensor
            let x_file = format!("data/{}", test_file);
            let x: ArrayD<i64> = read_npy(&x_file).unwrap();

            // Load expected output tensor
            let expected_file = format!("data/expand_for_batches_{}_result.npy", batch_size);
            let expected: ArrayD<i64> = read_npy(&expected_file).unwrap();

            // Call expand_for_batches function
            let result = expand_for_batches(&x, batch_size).unwrap();

            // Compare shapes
            assert_eq!(
                result.shape(),
                expected.shape(),
                "Shape mismatch for batch_size={}",
                batch_size
            );

            // Compare values
            assert_eq!(
                result, expected,
                "Value mismatch for batch_size={}",
                batch_size
            );
        } else {
            panic!("Invalid test file format: {}", test_file);
        }
    }
}
