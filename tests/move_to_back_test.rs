use RustOps::functions::move_to_back;
use ndarray::ArrayD;
use ndarray_npy::read_npy;

#[test]
fn test_move_to_back() {
    let original_file = "data/move_value_to_back_original.npy";
    let moved_file = "data/move_value_to_back_moved.npy";

    let original: ArrayD<i64> = read_npy(original_file).unwrap();
    let expected: ArrayD<i64> = read_npy(moved_file).unwrap();

    let result = move_to_back::move_to_back(&original, -2);

    assert_eq!(result, expected);
}
