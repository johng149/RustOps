use RustOps::functions::gather;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

const INPUT_PREFIX: &str = "gather";
const OUTPUT_SUFFIX: &str = "_gather_y.npy";
const INDEX_SUFFIX: &str = "_gather_indices.npy";
const INPUT_SUFFIX: &str = "_gather_x.npy";
const NPY_SUFFIX: &str = ".npy";

/// Parses the filename to extract gather dimensions.
/// Expected format: "gather<N>d_gather_x.npy"
fn parse_test_filename(filename: &str) -> Option<usize> {
    if !filename.starts_with(INPUT_PREFIX) || !filename.ends_with(INPUT_SUFFIX) {
        return None;
    }

    let dimension_part = filename
        .strip_prefix(INPUT_PREFIX)?
        .strip_suffix(INPUT_SUFFIX)?;

    // Extract the dimension number (e.g., "2d" -> 2)
    if let Some(dim_str) = dimension_part.strip_suffix('d') {
        dim_str.parse::<usize>().ok()
    } else {
        None
    }
}

/// Given directory path, discover all gather test files and return their paths.
fn discover_gather_test_files(dir: &str) -> Vec<String> {
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
fn test_gather_ndarray() {
    let test_files = discover_gather_test_files("data");

    for test_file in test_files {
        if let Some(dim) = parse_test_filename(&test_file) {
            // Extract the base name (e.g., "gather2d")
            let base_name = test_file.strip_suffix(INPUT_SUFFIX).unwrap();

            println!("Running test for {}, dimension: {}", base_name, dim);

            // Load input tensor
            let x_file = format!("data/{}", test_file);
            let x: ArrayD<f32> = read_npy(&x_file).unwrap();

            // Load indices tensor
            let indices_file = format!("data/{}_gather_indices{}", base_name, NPY_SUFFIX);
            let indices: ArrayD<i64> = read_npy(&indices_file).unwrap();

            // Load expected output tensor
            let expected_file = format!("data/{}_gather_y{}", base_name, NPY_SUFFIX);
            let expected: ArrayD<f32> = read_npy(&expected_file).unwrap();

            // Call gather function with last dimension (-1)
            let result = gather::gather(&x, -1, &indices).unwrap();

            assert_eq!(result, expected);
        } else {
            panic!("Invalid test file format: {}", test_file);
        }
    }
}
