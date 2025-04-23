use RustOps::functions::{
    argmaxi::argmaxi, expand_for_batches::expand_for_batches, growth_argmaxi::growth_argmaxi,
    mark_reserved_indices::mark_reserved_indices, move_to_back::move_to_back, sum::sum_generic,
};
use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, Axis, IxDyn};
use ndarray_npy::read_npy;

#[test]
fn test_growth_argmaxi() {
    // Load input files
    let xfile = "data/growth_argmaxi_x.npy";
    let countsfile = "data/growth_argmaxi_counts.npy";
    let x: ArrayD<f32> = read_npy(xfile).unwrap();
    let counts: ArrayD<i64> = read_npy(countsfile).unwrap();

    // Load expected output files
    let grown_file = "data/growth_argmaxi_grown.npy";
    let updated_counts_file = "data/growth_argmaxi_updated_counts.npy";
    let expected_grown: ArrayD<f32> = read_npy(grown_file).unwrap();
    let expected_updated_counts: ArrayD<f32> = read_npy(updated_counts_file).unwrap();
    let expected_updated_countsi64 = expected_updated_counts.mapv(|x| x as i64);

    // Parameters from the Python implementation
    let eps = 1e-8;
    let threshold = 0.5;
    let mark = -2;

    // Call the function
    let (grown, updated_counts) = growth_argmaxi(&x, &counts, eps, threshold, mark);

    // Assert results match expected outputs
    assert_abs_diff_eq!(grown, expected_grown, epsilon = 1e-5);
    assert_eq!(updated_counts, expected_updated_countsi64);
}
