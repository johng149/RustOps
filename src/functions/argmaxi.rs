use std::fmt::Debug;

use super::abs::{self, abs_ndarray};
use super::max::max;
use super::maxi::maxi;
use super::unsqueeze::unsqueeze;
use ndarray::{Array, ArrayD, Dimension, IxDyn, RemoveAxis, ScalarOperand};

pub fn argmaxi<T, D>(x: &Array<T, D>, epsilon: T) -> Array<T, D>
where
    T: num_traits::Float + Clone + Debug + ScalarOperand,
    D: Dimension + RemoveAxis,
{
    let maxi_result = maxi(x);
    let maxied = abs_ndarray(&maxi_result);
    let maxied_last_dim = maxied.ndim() - 1;

    // max returns both the values and the indices, but we only need the values
    let result = max(&maxied, maxied_last_dim).unwrap();
    let values = result.0;

    // element-wise subtraction between values and epsilon
    let subtracted = values - epsilon;

    let subtracted_last_dim = subtracted.ndim();
    let unsqueezed = unsqueeze(&subtracted.into_dyn(), subtracted_last_dim);

    let unsqueezed_d = unsqueezed.into_dimensionality::<D>().unwrap();

    // maxied should have shape (b, c, d) while unsqueezed_d should have shape (b, c, 1)
    maxied / unsqueezed_d
}
