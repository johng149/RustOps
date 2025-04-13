use RustOps::functions::wheres;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

// filepath: /media/john/Tertiary/Projects/ML/RustOps/tests/wheres_test.rs

#[test]
fn test_where_op() {
    let x_file = "data/where_where_x.npy";
    let condition_file = "data/where_condition.npy";
    let expected_result_file = "data/where_result.npy";

    // Read input tensors
    let x: ArrayD<f32> = read_npy(x_file).unwrap();
    let condition: ArrayD<bool> = read_npy(condition_file).unwrap();
    let expected_result: ArrayD<f32> = read_npy(expected_result_file).unwrap();

    // Create a zeros tensor for y (equivalent to torch.zeros_like(x) in the Python code)
    let y = ArrayD::<f32>::zeros(x.shape());

    // Call the where_op function
    let result = wheres::where_op(&condition, &x, &y).unwrap();

    // Assert the results match
    assert_eq!(result, expected_result);
}
