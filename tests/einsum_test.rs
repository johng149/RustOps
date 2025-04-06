use RustOps::functions::einsum;
use approx::assert_abs_diff_eq;
use ndarray::{Array, ArrayD, IxDyn};
use ndarray_npy::read_npy;

#[test]
fn test_einsum_ndarray_dyn_bfmd_bfd() {
    // Test matrix multiplication as einsum: 'batch fields memories dim, batch fields dim -> batch fields memories'
    let x_file = "data/einsum_batch_fields_memories_einsum_bfmd_bfd_x.npy";
    let y_file = "data/einsum_batch_fields_memories_einsum_bfmd_bfd_y.npy";
    let z_file = "data/einsum_batch_fields_memories_einsum_bfmd_bfd_z.npy";

    let x: ArrayD<f32> = read_npy(x_file).unwrap();
    let y: ArrayD<f32> = read_npy(y_file).unwrap();
    let z: ArrayD<f32> = read_npy(z_file).unwrap();

    let tensors = [&x, &y];
    let result = einsum::einsum_ndarray_dyn("bfmd,bfd->bfm", &tensors).unwrap();

    assert_abs_diff_eq!(result, z, epsilon = 1e-5);
}

#[test]
fn test_einsum_ndarray_dyn_bhchmc_bhcc() {
    // Test einsum: 'batch hidden children h_mems c_mems, batch hidden children c_mems -> batch hidden h_mems'
    // note that since h_mems, hidden start with the same letter, we will use h for h_mems and c for c_mems
    // u for hidden, and k for children
    let x_file = "data/einsum_batch_hidden_children_mems_einsum_bhchmc_bhcc_x.npy";
    let y_file = "data/einsum_batch_hidden_children_mems_einsum_bhchmc_bhcc_y.npy";
    let z_file = "data/einsum_batch_hidden_children_mems_einsum_bhchmc_bhcc_z.npy";

    let x: ArrayD<f32> = read_npy(x_file).unwrap();
    let y: ArrayD<f32> = read_npy(y_file).unwrap();
    let z: ArrayD<f32> = read_npy(z_file).unwrap();

    let tensors = [&x, &y];
    let result = einsum::einsum_ndarray_dyn("bukhc,bukc->buh", &tensors).unwrap();

    assert_abs_diff_eq!(result, z, epsilon = 1e-5);
}

#[test]
fn test_einsum_ndarray_dyn_bpcp_bpcpc() {
    // Test einsum: 'batch parents children pdim, batch parents children pdim cdim -> batch parents children cdim
    // note that since pdim, parents start with the same letter, we will use p for parent and d for pdim
    // c for children, and k for cdim
    let x_file = "data/einsum_batch_parents_children_pdim_cdim_einsum_bpcp_bpcpc_x.npy";
    let y_file = "data/einsum_batch_parents_children_pdim_cdim_einsum_bpcp_bpcpc_y.npy";
    let z_file = "data/einsum_batch_parents_children_pdim_cdim_einsum_bpcp_bpcpc_z.npy";
    let x: ArrayD<f32> = read_npy(x_file).unwrap();
    let y: ArrayD<f32> = read_npy(y_file).unwrap();
    let z: ArrayD<f32> = read_npy(z_file).unwrap();
    let tensors = [&x, &y];
    let result = einsum::einsum_ndarray_dyn("bpcd,bpcdk->bpck", &tensors).unwrap();
    assert_abs_diff_eq!(result, z, epsilon = 1e-5);
}

#[test]
fn test_einsum_ndarray_dyn_ncmd_bnm() {
    // Test einsum: 'nodes children_per_node memories dim, batch nodes memories -> batch nodes children_per_node dim'
    let x_file = "data/einsum_nodes_children_memories_dim_einsum_ncmd_bnm_x.npy";
    let y_file = "data/einsum_nodes_children_memories_dim_einsum_ncmd_bnm_y.npy";
    let z_file = "data/einsum_nodes_children_memories_dim_einsum_ncmd_bnm_z.npy";
    let x: ArrayD<f32> = read_npy(x_file).unwrap();
    let y: ArrayD<f32> = read_npy(y_file).unwrap();
    let z: ArrayD<f32> = read_npy(z_file).unwrap();
    let tensors = [&x, &y];
    let result = einsum::einsum_ndarray_dyn("ncmd,bnm->bncd", &tensors).unwrap();
    assert_abs_diff_eq!(result, z, epsilon = 1e-5);
}

#[test]
fn test_einsum_ndarray_dyn_bncd_bnm() {
    // Test einsum: 'batch nodes children_per_node dim, batch nodes memories -> nodes children_per_node memories dim'
    let x_file = "data/einsum_batch_nodes_children_dim_memories_einsum_bncd_bnm_x.npy";
    let y_file = "data/einsum_batch_nodes_children_dim_memories_einsum_bncd_bnm_y.npy";
    let z_file = "data/einsum_batch_nodes_children_dim_memories_einsum_bncd_bnm_z.npy";
    let x: ArrayD<f32> = read_npy(x_file).unwrap();
    let y: ArrayD<f32> = read_npy(y_file).unwrap();
    let z: ArrayD<f32> = read_npy(z_file).unwrap();
    let tensors = [&x, &y];
    let result = einsum::einsum_ndarray_dyn("bncd,bnm->ncmd", &tensors).unwrap();
    assert_abs_diff_eq!(result, z, epsilon = 1e-5);
}
