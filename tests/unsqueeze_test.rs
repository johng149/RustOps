use RustOps::functions::unsqueeze;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/unsqueeze_test.rs

const OUTPUT_PREFIX: &str = "unsqueeze";
const NPY_SUFFIX: &str = ".npy";

/// Parses the filename to extract unsqueeze dimension parameter.
/// Expected format: "unsqueeze<N>d_unsqueeze_y_dim<D>.npy"
fn parse_test_filename(filename: &str) -> Option<(String, usize)> {
    if !filename.ends_with(NPY_SUFFIX) {
        return None;
    }

    // Extract parts like "unsqueeze2d_unsqueeze_y_dim0"
    let core = filename.strip_suffix(NPY_SUFFIX)?;
    if !core.contains("_unsqueeze_y_dim") {
        return None;
    }

    // Split into base name and dimension
    let parts: Vec<&str> = core.split("_unsqueeze_y_dim").collect();
    if parts.len() != 2 {
        return None;
    }

    let base_name = parts[0].to_string();
    let dim = parts[1].parse::<usize>().ok()?;

    Some((base_name, dim))
}

/// Given directory path, discover all unsqueeze test files and return their paths.
fn discover_unsqueeze_test_files(dir: &str) -> Vec<String> {
    let mut test_files = Vec::new();
    let entries = std::fs::read_dir(dir).unwrap();

    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_file() {
            if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                if filename.contains("_unsqueeze_y_dim") && filename.ends_with(NPY_SUFFIX) {
                    test_files.push(filename.to_string());
                }
            }
        }
    }

    test_files
}

#[test]
fn test_unsqueeze_ndarray() {
    let test_files = discover_unsqueeze_test_files("data");

    for test_file in test_files {
        if let Some((base_name, dim)) = parse_test_filename(&test_file) {
            println!("Running test for file: {}, dim: {}", test_file, dim);

            // Load input tensor
            let xfile = format!("data/{}_unsqueeze_x.npy", base_name);
            let x: ArrayD<f32> = read_npy(&xfile).unwrap();

            // Load expected output
            let yfile = format!("data/{}", test_file);
            let expected: ArrayD<f32> = read_npy(&yfile).unwrap();

            // Apply unsqueeze operation
            let result = unsqueeze::unsqueeze(&x, dim);

            // Assert equality
            assert_eq!(result, expected);

            // Assert shapes are equal
            assert_eq!(
                result.shape(),
                expected.shape(),
                "Shape mismatch for {}. Got {:?}, expected {:?}",
                test_file,
                result.shape(),
                expected.shape()
            );
        } else {
            panic!("Invalid test file format: {}", test_file);
        }
    }
}
