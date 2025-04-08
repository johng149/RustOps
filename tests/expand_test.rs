use RustOps::functions::expand;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/expand_test.rs

const NAME_PREFIXES: [&str; 4] = ["expand2d", "expand3d", "expand4d", "expand5d"];
const ORIGINAL_SUFFIX: &str = "_original_expand_dim.npy";
const EXPANDED_PREFIX: &str = "_expand_dim_";
const NPY_SUFFIX: &str = ".npy";

/// Parses the filename to extract expand parameters.
/// Expected format: "expand{dimension}d_expand_dim_{N}.npy"
fn parse_test_filename(filename: &str) -> Option<(String, usize)> {
    for prefix in &NAME_PREFIXES {
        let expanded_str = format!("{}{}", prefix, EXPANDED_PREFIX);
        if filename.starts_with(&expanded_str) && filename.ends_with(NPY_SUFFIX) {
            let dim_part = filename
                .strip_prefix(&expanded_str)?
                .strip_suffix(NPY_SUFFIX)?;

            if let Ok(dim) = dim_part.parse::<usize>() {
                return Some((prefix.to_string(), dim));
            }
        }
    }
    None
}

/// Given directory path, discover all expand test files and return their paths.
fn discover_expand_test_files(dir: &str) -> Vec<(String, String, usize)> {
    let mut test_files = Vec::new();
    let entries = std::fs::read_dir(dir).unwrap();

    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_file() {
            if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                if let Some((base_name, dim)) = parse_test_filename(filename) {
                    let original_file = format!("{}{}", base_name, ORIGINAL_SUFFIX);
                    test_files.push((base_name, original_file, dim));
                }
            }
        }
    }

    test_files.sort();
    test_files.dedup();
    test_files
}

#[test]
fn test_expand_ndarray() {
    let test_cases = discover_expand_test_files("data");

    for (base_name, original_file, dim) in test_cases {
        println!("Testing expansion for {}, dim: {}", base_name, dim);

        // Load the original tensor
        let x_file = format!("data/{}", original_file);
        let x: ArrayD<f32> = read_npy(&x_file).unwrap();

        // Load the expected expanded tensor
        let expected_file = format!("data/{}{}{}{}", base_name, EXPANDED_PREFIX, dim, NPY_SUFFIX);
        let expected: ArrayD<f32> = read_npy(&expected_file).unwrap();

        // Get the expansion size from the expected tensor shape
        let size = expected.shape()[dim];

        // Perform the expansion
        let result = expand::expand_at_dim(&x, dim, size).unwrap();

        // Verify the result
        assert_eq!(result.shape(), expected.shape());
        assert_eq!(result, expected);
    }
}
