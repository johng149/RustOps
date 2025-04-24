import torch
import einops
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_tensors(
    batch_size: int,
    nodes: int,
    children_per_node: int,
    memories: int,
    dim: int,
    dtype: torch.dtype = torch.float32,
    parent_down_prop = None,
    parent_mem_matrix = None
):
    # Create random tensors if not provided
    if parent_down_prop is None:
        parent_down_prop = torch.rand((batch_size, nodes, memories), dtype=dtype)
    if parent_mem_matrix is None:
        parent_mem_matrix = torch.rand((nodes, children_per_node, memories, dim), dtype=dtype)
    
    # Expand parent_mem_matrix if it's 3D
    mm = parent_mem_matrix if parent_mem_matrix.dim() != 3 else parent_mem_matrix.unsqueeze(-3)
    
    # Execute the pred logic
    prediction = einops.einsum(
        mm, parent_down_prop, 
        'nodes children_per_node memories dim, batch nodes memories -> batch nodes children_per_node dim'
    )
    prediction = prediction.reshape(batch_size, nodes * children_per_node, dim)
    
    return prediction, parent_down_prop, parent_mem_matrix

def create_pred_reference(
    batch_size: int,
    nodes: int,
    children_per_node: int,
    memories: int,
    dim: int,
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "pred",
):
    prediction, parent_down_prop, parent_mem_matrix = create_tensors(
        batch_size, nodes, children_per_node, memories, dim, dtype
    )
    
    save_reference(prediction, dir, f"{name}_prediction")
    save_reference(parent_down_prop, dir, f"{name}_parent_down_prop")
    save_reference(parent_mem_matrix, dir, f"{name}_parent_mem_matrix")

if __name__ == "__main__":
    create_pred_reference(
        batch_size=6,
        nodes=4,
        children_per_node=3,
        memories=8,
        dim=5,
        dtype=torch.float32,
        dir="data",
        name="pred"
    )