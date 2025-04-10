use RustOps::functions::scatter::scatter;
use approx::assert_abs_diff_eq;
use ndarray::{Array, ArrayD, Dimension, IxDyn};
use ndarray_npy::read_npy;

#[test]
fn test_rearrange_matches_reference() {
    let indices = "data/scatter2d_scatter_indices.npy";
    let gathered = "data/scatter2d_scatter_gathered.npy";
    let blank = "data/scatter2d_scatter_blank.npy";
    let scattered = "data/scatter2d_scatter_scattered.npy";

    let i: ArrayD<i64> = read_npy(indices).unwrap();
    let g: ArrayD<f32> = read_npy(gathered).unwrap();
    let mut b: ArrayD<f32> = read_npy(blank).unwrap();
    let s: ArrayD<f32> = read_npy(scattered).unwrap();

    scatter(&mut b, 1, &i, &g);

    assert_abs_diff_eq!(b, s, epsilon = 1e-5);
}

#[test]
fn test_scatter3d_matches_reference() {
    let indices = "data/scatter3d_scatter_indices.npy";
    let gathered = "data/scatter3d_scatter_gathered.npy";
    let blank = "data/scatter3d_scatter_blank.npy";
    let scattered = "data/scatter3d_scatter_scattered.npy";

    let i: ArrayD<i64> = read_npy(indices).unwrap();
    let g: ArrayD<f32> = read_npy(gathered).unwrap();
    let mut b: ArrayD<f32> = read_npy(blank).unwrap();
    let s: ArrayD<f32> = read_npy(scattered).unwrap();

    scatter(&mut b, 2, &i, &g);

    assert_abs_diff_eq!(b, s, epsilon = 1e-5);
}

#[test]
fn test_scatter4d_matches_reference() {
    let indices = "data/scatter4d_scatter_indices.npy";
    let gathered = "data/scatter4d_scatter_gathered.npy";
    let blank = "data/scatter4d_scatter_blank.npy";
    let scattered = "data/scatter4d_scatter_scattered.npy";

    let i: ArrayD<i64> = read_npy(indices).unwrap();
    let g: ArrayD<f32> = read_npy(gathered).unwrap();
    let mut b: ArrayD<f32> = read_npy(blank).unwrap();
    let s: ArrayD<f32> = read_npy(scattered).unwrap();

    scatter(&mut b, 3, &i, &g);

    assert_abs_diff_eq!(b, s, epsilon = 1e-5);
}

#[test]
fn test_scatter5d_matches_reference() {
    let indices = "data/scatter5d_scatter_indices.npy";
    let gathered = "data/scatter5d_scatter_gathered.npy";
    let blank = "data/scatter5d_scatter_blank.npy";
    let scattered = "data/scatter5d_scatter_scattered.npy";

    let i: ArrayD<i64> = read_npy(indices).unwrap();
    let g: ArrayD<f32> = read_npy(gathered).unwrap();
    let mut b: ArrayD<f32> = read_npy(blank).unwrap();
    let s: ArrayD<f32> = read_npy(scattered).unwrap();

    scatter(&mut b, 4, &i, &g);

    assert_abs_diff_eq!(b, s, epsilon = 1e-5);
}

#[test]
fn test_scatter6d_matches_reference() {
    let indices = "data/scatter6d_scatter_indices.npy";
    let gathered = "data/scatter6d_scatter_gathered.npy";
    let blank = "data/scatter6d_scatter_blank.npy";
    let scattered = "data/scatter6d_scatter_scattered.npy";

    let i: ArrayD<i64> = read_npy(indices).unwrap();
    let g: ArrayD<f32> = read_npy(gathered).unwrap();
    let mut b: ArrayD<f32> = read_npy(blank).unwrap();
    let s: ArrayD<f32> = read_npy(scattered).unwrap();

    scatter(&mut b, 5, &i, &g);

    assert_abs_diff_eq!(b, s, epsilon = 1e-5);
}

#[test]
fn test_scatter7d_matches_reference() {
    let indices = "data/scatter7d_scatter_indices.npy";
    let gathered = "data/scatter7d_scatter_gathered.npy";
    let blank = "data/scatter7d_scatter_blank.npy";
    let scattered = "data/scatter7d_scatter_scattered.npy";

    let i: ArrayD<i64> = read_npy(indices).unwrap();
    let g: ArrayD<f32> = read_npy(gathered).unwrap();
    let mut b: ArrayD<f32> = read_npy(blank).unwrap();
    let s: ArrayD<f32> = read_npy(scattered).unwrap();

    scatter(&mut b, 6, &i, &g);

    assert_abs_diff_eq!(b, s, epsilon = 1e-5);
}
