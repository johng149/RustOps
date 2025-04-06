import torch
from einops import einsum
from util.save_reference import save_reference
from typing import Tuple, List, Iterable


def test_einsum_batch_fields_memories(
    shape1: Tuple[int, ...] | List[int] | Iterable[int],
    shape2: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "einsum_batch_fields_memories",
):
    """
    Test einsum operation `einops.einsum('batch fields memories dim, batch fields dim -> batch fields memories')` on tensors.
    Args:
        shape1 (Tuple[int, ...]): Shape of the first tensor to create.
        shape2 (Tuple[int, ...]): Shape of the second tensor to create.
        dtype (torch.dtype): Data type of the tensors. Default is torch.float32.
        dir (str): Directory to save the reference tensors. Default is "data".
        name (str): Name of the reference tensor file. Default is "einsum_batch_fields_memories".
    """
    x = torch.rand(shape1, dtype=dtype)
    y = torch.rand(shape2, dtype=dtype)
    z = einsum(x, y, 'batch fields memories dim, batch fields dim -> batch fields memories')
    # Save the original tensors
    save_reference(x, dir, f"{name}_einsum_bfmd_bfd_x")
    save_reference(y, dir, f"{name}_einsum_bfmd_bfd_y")
    # Save the resulting tensor
    save_reference(z, dir, f"{name}_einsum_bfmd_bfd_z")

def test_einsum_batch_hidden_children_mems(
    shape1: Tuple[int, ...] | List[int] | Iterable[int],
    shape2: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "einsum_batch_hidden_children_mems",
):
    """
    Test einsum operation `einops.einsum('batch hidden children h_mems c_mems, batch hidden children c_mems -> batch hidden h_mems')` on tensors.
    Args:
        shape1 (Tuple[int, ...]): Shape of the first tensor to create.
        shape2 (Tuple[int, ...]): Shape of the second tensor to create.
        dtype (torch.dtype): Data type of the tensors. Default is torch.float32.
        dir (str): Directory to save the reference tensors. Default is "data".
        name (str): Name of the reference tensor file. Default is "einsum_batch_hidden_children_mems".
    """
    x = torch.rand(shape1, dtype=dtype)
    y = torch.rand(shape2, dtype=dtype)
    z = einsum(x, y, 'batch hidden children h_mems c_mems, batch hidden children c_mems -> batch hidden h_mems')
    # Save the original tensors
    save_reference(x, dir, f"{name}_einsum_bhchmc_bhcc_x")
    save_reference(y, dir, f"{name}_einsum_bhchmc_bhcc_y")
    # Save the resulting tensor
    save_reference(z, dir, f"{name}_einsum_bhchmc_bhcc_z")

def test_einsum_batch_parents_children_pdim_cdim(
    shape1: Tuple[int, ...] | List[int] | Iterable[int],
    shape2: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "einsum_batch_parents_children_pdim_cdim",
):
    """
    Test einsum operation `einops.einsum('batch parents children pdim, batch parents children pdim cdim -> batch parents children cdim')` on tensors.
    Args:
        shape1 (Tuple[int, ...]): Shape of the first tensor to create.
        shape2 (Tuple[int, ...]): Shape of the second tensor to create.
        dtype (torch.dtype): Data type of the tensors. Default is torch.float32.
        dir (str): Directory to save the reference tensors. Default is "data".
        name (str): Name of the reference tensor file. Default is "einsum_batch_parents_children_pdim_cdim".
    """
    x = torch.rand(shape1, dtype=dtype)
    y = torch.rand(shape2, dtype=dtype)
    z = einsum(x, y, 'batch parents children pdim, batch parents children pdim cdim -> batch parents children cdim')
    # Save the original tensors
    save_reference(x, dir, f"{name}_einsum_bpcp_bpcpc_x")
    save_reference(y, dir, f"{name}_einsum_bpcp_bpcpc_y")
    # Save the resulting tensor
    save_reference(z, dir, f"{name}_einsum_bpcp_bpcpc_z")

def test_einsum_nodes_children_memories_dim(
    shape1: Tuple[int, ...] | List[int] | Iterable[int],
    shape2: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "einsum_nodes_children_memories_dim",
):
    """
    Test einsum operation `einops.einsum('nodes children_per_node memories dim, batch nodes memories -> batch nodes children_per_node dim')` on tensors.
    Args:
        shape1 (Tuple[int, ...]): Shape of the first tensor to create.
        shape2 (Tuple[int, ...]): Shape of the second tensor to create.
        dtype (torch.dtype): Data type of the tensors. Default is torch.float32.
        dir (str): Directory to save the reference tensors. Default is "data".
        name (str): Name of the reference tensor file. Default is "einsum_nodes_children_memories_dim".
    """
    x = torch.rand(shape1, dtype=dtype)
    y = torch.rand(shape2, dtype=dtype)
    z = einsum(x, y, 'nodes children_per_node memories dim, batch nodes memories -> batch nodes children_per_node dim')
    # Save the original tensors
    save_reference(x, dir, f"{name}_einsum_ncmd_bnm_x")
    save_reference(y, dir, f"{name}_einsum_ncmd_bnm_y")
    # Save the resulting tensor
    save_reference(z, dir, f"{name}_einsum_ncmd_bnm_z")

def test_einsum_batch_nodes_children_dim_memories(
    shape1: Tuple[int, ...] | List[int] | Iterable[int],
    shape2: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "einsum_batch_nodes_children_dim_memories",
):
    """
    Test einsum operation `einops.einsum('batch nodes children_per_node dim, batch nodes memories -> nodes children_per_node memories dim')` on tensors.
    Args:
        shape1 (Tuple[int, ...]): Shape of the first tensor to create.
        shape2 (Tuple[int, ...]): Shape of the second tensor to create.
        dtype (torch.dtype): Data type of the tensors. Default is torch.float32.
        dir (str): Directory to save the reference tensors. Default is "data".
        name (str): Name of the reference tensor file. Default is "einsum_batch_nodes_children_dim_memories".
    """
    x = torch.rand(shape1, dtype=dtype)
    y = torch.rand(shape2, dtype=dtype)
    z = einsum(x, y, 'batch nodes children_per_node dim, batch nodes memories -> nodes children_per_node memories dim')
    # Save the original tensors
    save_reference(x, dir, f"{name}_einsum_bncd_bnm_x")
    save_reference(y, dir, f"{name}_einsum_bncd_bnm_y")
    # Save the resulting tensor
    save_reference(z, dir, f"{name}_einsum_bncd_bnm_z")

if __name__ == "__main__":
    test_einsum_batch_fields_memories(
        (2, 3, 4, 5), (2, 3, 5), dtype=torch.float32, dir="data", name="einsum_batch_fields_memories"
    )
    test_einsum_batch_hidden_children_mems(
        (2, 3, 4, 5, 6), (2, 3, 4, 6), dtype=torch.float32, dir="data", name="einsum_batch_hidden_children_mems"
    )
    test_einsum_batch_parents_children_pdim_cdim(
        (2, 3, 4, 5), (2, 3, 4, 5, 6), dtype=torch.float32, dir="data", name="einsum_batch_parents_children_pdim_cdim"
    )
    test_einsum_nodes_children_memories_dim(
        (2, 3, 4, 5), (6, 2, 4), dtype=torch.float32, dir="data", name="einsum_nodes_children_memories_dim"
    )
    test_einsum_batch_nodes_children_dim_memories(
        (2, 3, 4, 5), (2, 3, 6), dtype=torch.float32, dir="data", name="einsum_batch_nodes_children_dim_memories"
    )