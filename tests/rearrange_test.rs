use RustOps::functions::rearrange::rearrange_batch_mems_flag;
use approx::assert_abs_diff_eq;
use ndarray::{Array, ArrayD, Dimension, IxDyn};
use ndarray_npy::read_npy;

#[test]
fn test_rearrange_matches_reference() {
    let xfile = "data/rearranged_rearrange_original.npy";
    let yfile = "data/rearranged_rearrange_result.npy";
    let original: ArrayD<f32> = read_npy(xfile).unwrap();
    let expected: ArrayD<f32> = read_npy(yfile).unwrap();

    // Apply our Rust implementation
    let result = rearrange_batch_mems_flag(&original).unwrap();

    // Print the shapes for debugging
    println!("Original shape: {:?}", original.shape());
    println!("Result shape: {:?}", result.shape());
    println!("Expected shape: {:?}", expected.shape());

    // Verify shapes match
    assert_eq!(result.shape(), expected.shape());

    // Verify values match within a small epsilon
    assert_abs_diff_eq!(result, expected, epsilon = 1e-5);
}
