import torch
import einops
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
from pred import create_tensors


def mem_delta(parent_down_prop, parent_mem_matrix, child_down_prop):
    """
    Calculates the delta that should be added to the given `parent_mem_matrix`
    to optimize the model. For example, to optimize the given matrix, use
    `new_mm = old_mm + lr * mem_delta(p_prop, old_mm, c_prop)`
    Note that the `lr` scaling should be done outside of this function

    Note also that, according to the paper / reference implementation, when
    calculating the matrix update for the outer layer, the child down prop is
    just the raw inputs, not one that is one-hot encoded
    """
    # shape is nodes memories dim, but we want
    # nodes children_per_node memories dim
    # in this case, we'll assume that children_per_node is 1
    mm = parent_mem_matrix if parent_mem_matrix.dim() != 3 else parent_mem_matrix.unsqueeze(-3)
    nodes, children_per_node, memories, dim = mm.shape
    batch_size, nodes, memories = parent_down_prop.shape

    # Use create_tensors to get prediction with the same logic as pred
    prediction, _, _ = create_tensors(
        batch_size=batch_size,
        nodes=nodes,
        children_per_node=children_per_node,
        memories=memories,
        dim=dim,
        dtype=parent_down_prop.dtype,
        parent_down_prop=parent_down_prop,
        parent_mem_matrix=parent_mem_matrix
    )

    error = child_down_prop - prediction
    error = error.reshape(batch_size, nodes, children_per_node, dim)
    delta = einops.einsum(error, parent_down_prop, 'batch nodes children_per_node dim, batch nodes memories -> nodes children_per_node memories dim')
    return delta.reshape(parent_mem_matrix.shape)

def create_mem_delta_tensors(
    batch_size: int,
    nodes: int,
    children_per_node: int,
    memories: int,
    dim: int,
    dtype: torch.dtype = torch.float32,
):
    # Create initial tensors using the function from pred.py
    prediction, parent_down_prop, parent_mem_matrix = create_tensors(
        batch_size, nodes, children_per_node, memories, dim, dtype
    )
    
    # Create a random child_down_prop tensor
    child_down_prop = torch.rand_like(prediction)
    
    # Calculate mem_delta
    delta = mem_delta(parent_down_prop, parent_mem_matrix, child_down_prop)
    
    return delta, parent_down_prop, parent_mem_matrix, child_down_prop

def create_mem_delta_reference(
    batch_size: int,
    nodes: int,
    children_per_node: int,
    memories: int,
    dim: int,
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "mem_delta",
):
    delta, parent_down_prop, parent_mem_matrix, child_down_prop = create_mem_delta_tensors(
        batch_size, nodes, children_per_node, memories, dim, dtype
    )
    
    save_reference(delta, dir, f"{name}_delta")
    save_reference(parent_down_prop, dir, f"{name}_parent_down_prop")
    save_reference(parent_mem_matrix, dir, f"{name}_parent_mem_matrix")
    save_reference(child_down_prop, dir, f"{name}_child_down_prop")

if __name__ == "__main__":
    create_mem_delta_reference(
        batch_size=6,
        nodes=4,
        children_per_node=3,
        memories=8,
        dim=5,
        dtype=torch.float32,
        dir="data",
        name="mem_delta"
    )