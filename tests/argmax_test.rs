use RustOps::functions::argmax;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

const INPUT_FILENAME: &str = "argmax_argmax_x.npy";
const OUTPUT_PREFIX: &str = "argmaxargmax_";
const NPY_SUFFIX: &str = ".npy";

/// Parses the filename to extract argmax parameters.
/// Expected format: "argmaxargmax_dim<N>_<yes|no>keepdim.npy" or "argmaxargmax_flattened.npy"
fn parse_test_filename(filename: &str) -> Option<(Option<usize>, bool)> {
    if !filename.starts_with(OUTPUT_PREFIX) || !filename.ends_with(NPY_SUFFIX) {
        return None;
    }

    let core = filename
        .strip_prefix(OUTPUT_PREFIX)?
        .strip_suffix(NPY_SUFFIX)?;

    if core == "flattened" {
        // Convention for dim=None. Keepdim doesn't matter for flattened.
        return Some((None, false));
    }

    let parts: Vec<&str> = core.split('_').collect();
    if parts.len() != 2 {
        return None; // Expecting dim<N>_keepdim format
    }

    // Part 1: Dimension
    let dim_part = parts[0];
    let dim = if dim_part.starts_with("dim") {
        dim_part.strip_prefix("dim")?.parse::<usize>().ok()
    } else {
        None // Invalid dimension format
    };

    // Part 2: Keepdim
    let keepdim_part = parts[1];
    let keepdim = match keepdim_part {
        "yeskeepdim" => Some(true),
        "nokeepdim" => Some(false),
        _ => None, // Invalid keepdim format
    };

    // We need both dim (or knowledge it's None) and keepdim to be valid
    match (dim, keepdim) {
        (Some(d), Some(k)) => Some((Some(d), k)),
        _ => None, // If either part failed parsing
    }
}

/// Given directory path, discover all argmax test files and return their paths.
fn discover_argmax_test_files(dir: &str) -> Vec<String> {
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
fn test_argmax_ndarray() {
    let test_files = discover_argmax_test_files("data");
    let xfile = format!("data/{}", INPUT_FILENAME);
    let x: ArrayD<f32> = read_npy(&xfile).unwrap();

    for test_file in test_files {
        if let Some((dim, keepdim)) = parse_test_filename(&test_file) {
            // print out the test file name
            println!(
                "Running test for file: {}, dim: {:?}, keepdim: {}",
                test_file, dim, keepdim
            );
            let yfile = format!("data/{}", test_file);
            let y: ArrayD<i64> = read_npy(&yfile).unwrap();

            let result = argmax::argmax(&x, dim, keepdim).unwrap();
            assert_eq!(result, y);
        } else {
            panic!("Invalid test file format: {}", test_file);
        }
    }
}
