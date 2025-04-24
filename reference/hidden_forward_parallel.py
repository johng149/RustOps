from maxi import create_tensors as create_tensors_maxi
import torch
import einops
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_tensors(
    batch_size: int,
    num_hidden_nodes: int,
    children_per_hidden: int,
    hidden_mems: int,
    children_mem_cols: int,
    rho: float = 1e-8,
    dtype: torch.dtype = torch.float32,
    hidden_mm = None,
    children_x = None
):
    # Create random tensors if not provided
    if hidden_mm is None:
        hidden_mm = torch.rand((num_hidden_nodes, children_per_hidden, hidden_mems, children_mem_cols), dtype=dtype)
    if children_x is None:
        children_x = torch.rand((batch_size, num_hidden_nodes * children_per_hidden, children_mem_cols), dtype=dtype)
    
    # Expand hidden_mm if it's 4D
    if hidden_mm.dim() == 4:
        mm = hidden_mm.expand(batch_size, -1, -1, -1, -1)
    else:
        mm = hidden_mm
    
    # Apply maxi function to children_x
    _, max_indices, _, _, scattered = create_tensors_maxi(children_x.shape, dtype=children_x.dtype, x=children_x)
    x = scattered.reshape(batch_size, num_hidden_nodes, children_per_hidden, children_mem_cols)
    
    # Execute the hidden_forward_parallel logic
    propagation = einops.einsum(mm, x, 'batch hidden children h_mems c_mems, batch hidden children c_mems -> batch hidden h_mems')
    x_norm = torch.sqrt(einops.reduce(x**2,'batch hidden children c_mems -> batch hidden', 'sum')).unsqueeze(-1)
    norm_coeff = 1 / ((children_per_hidden * x_norm) + rho)
    result = propagation * norm_coeff
    
    return result, hidden_mm, children_x, propagation, x_norm, norm_coeff, x

def create_hidden_forward_parallel(
    batch_size: int,
    num_hidden_nodes: int,
    children_per_hidden: int,
    hidden_mems: int,
    children_mem_cols: int,
    rho: float = 1e-8,
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "hidden_forward_parallel",
):
    result, hidden_mm, children_x, propagation, x_norm, norm_coeff, x = create_tensors(
        batch_size, num_hidden_nodes, children_per_hidden, hidden_mems, children_mem_cols, rho, dtype
    )
    
    save_reference(result, dir, f"{name}_result_rho{rho}")
    save_reference(hidden_mm, dir, f"{name}_hidden_mm_rho{rho}")
    save_reference(children_x, dir, f"{name}_children_x_rho{rho}")
    save_reference(propagation, dir, f"{name}_propagation_rho{rho}")
    save_reference(x_norm, dir, f"{name}_x_norm_rho{rho}")
    save_reference(norm_coeff, dir, f"{name}_norm_coeff_rho{rho}")
    save_reference(x, dir, f"{name}_x_rho{rho}")

if __name__ == "__main__":
    create_hidden_forward_parallel(
        batch_size=6,
        num_hidden_nodes=4,
        children_per_hidden=3,
        hidden_mems=8,
        children_mem_cols=5,
        rho=1e-8,
        dtype=torch.float32,
        dir="data",
        name="hidden_forward_parallel"
    )