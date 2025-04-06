use ndarray::prelude::*;
use ndarray_einsum_beta::*;

fn main() {
    let m1 = arr1(&[1, 2]);
    let m2 = arr2(&[[1, 2], [3, 4]]);
    println!("{:?}", einsum("i,ij->j", &[&m1, &m2]));
}
