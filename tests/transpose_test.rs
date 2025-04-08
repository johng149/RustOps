// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/transpose_test.rs
use RustOps::functions::transpose;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

const INPUT_FILENAME: &str = "transpose_transpose_x.npy";
const OUTPUT_PREFIX: &str = "transpose_transpose_";
const NPY_SUFFIX: &str = ".npy";

/// Parses the filename to extract transpose parameters.
/// Expected format: "transpose_transpose_<i>_<i+1>.npy"
fn parse_test_filename(filename: &str) -> Option<(usize, usize)> {
    if !filename.starts_with(OUTPUT_PREFIX) || !filename.ends_with(NPY_SUFFIX) {
        return None;
    }

    let core = filename
        .strip_prefix(OUTPUT_PREFIX)?
        .strip_suffix(NPY_SUFFIX)?;

    let parts: Vec<&str> = core.split('_').collect();
    if parts.len() != 2 {
        return None; // Expecting <i>_<i+1> format
    }

    // Parse dimensions
    let dim1 = parts[0].parse::<usize>().ok()?;
    let dim2 = parts[1].parse::<usize>().ok()?;

    Some((dim1, dim2))
}

/// Given directory path, discover all transpose test files and return their paths.
fn discover_transpose_test_files(dir: &str) -> Vec<String> {
    let mut test_files = Vec::new();
    let entries = std::fs::read_dir(dir).unwrap();

    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_file() {
            if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                if filename.starts_with(OUTPUT_PREFIX)
                    && filename.ends_with(NPY_SUFFIX)
                    && filename != INPUT_FILENAME
                {
                    // Skip the input file
                    test_files.push(filename.to_string());
                }
            }
        }
    }

    test_files
}

#[test]
fn test_transpose_ndarray() {
    let test_files = discover_transpose_test_files("data");
    let xfile = format!("data/{}", INPUT_FILENAME);
    let x: ArrayD<f32> = read_npy(&xfile).unwrap();

    for test_file in test_files {
        if let Some((dim1, dim2)) = parse_test_filename(&test_file) {
            println!(
                "Running test for file: {}, dim1: {}, dim2: {}",
                test_file, dim1, dim2
            );
            let yfile = format!("data/{}", test_file);
            let y: ArrayD<f32> = read_npy(&yfile).unwrap();

            let result = transpose::transpose_dims(&x, dim1, dim2);
            assert_eq!(result, y);
        } else {
            panic!("Invalid test file format: {}", test_file);
        }
    }
}
