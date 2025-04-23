use RustOps::functions::sum;
use ndarray::{Array2, Array3, ArrayD, Axis};
use ndarray_npy::read_npy;

#[test]
fn test_sum_ndarray_3d() {
    let xfile = "data/sum_3d_x.npy";
    let yfile = "data/sum_3d_y.npy";
    let x: ArrayD<bool> = read_npy(xfile).unwrap();
    let y: ArrayD<i64> = read_npy(yfile).unwrap();

    // In the Python code, sum is computed along the last axis (-1) with keepdim=True
    let result = sum::sum_generic(&x.mapv(|v| v as i64), Axis(2), true);

    // Convert result back to i64 for comparison with expected output
    let result_i64 = result.mapv(|v| v as i64);

    assert_eq!(result_i64, y);
}
