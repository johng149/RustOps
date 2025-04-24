import torch
import einops
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_tensors(
    batch_size: int,
    parent_nodes: int,
    children_per_parent: int,
    parent_dim: int,
    child_dim: int,
    coeff: float = 0.5,
    dtype: torch.dtype = torch.float32,
    parent_h = None,
    parent_mm = None,
    child_h = None
):
    # Create random tensors if not provided
    if parent_h is None:
        parent_h = torch.rand((batch_size, parent_nodes, parent_dim), dtype=dtype)
    if parent_mm is None:
        parent_mm = torch.rand((parent_nodes, children_per_parent, parent_dim, child_dim), dtype=dtype)
    if child_h is None:
        child_h = torch.rand((batch_size, parent_nodes * children_per_parent, child_dim), dtype=dtype)
    
    # Expand parent_mm if it's 4D
    if parent_mm.dim() == 4:
        mm = parent_mm.expand(batch_size, -1, -1, -1, -1)
    else:
        mm = parent_mm
    
    # Execute the down_prop_parallel logic
    argmaxi_parent_h = parent_h.unsqueeze(-2).expand(-1, -1, children_per_parent, -1)
    orig = child_h * coeff
    new = (1 - coeff) * einops.einsum(
        argmaxi_parent_h, mm, 
        'batch parents children pdim, batch parents children pdim cdim -> batch parents children cdim'
    )
    new = new.reshape(batch_size, parent_nodes * children_per_parent, child_dim)
    result = orig + new
    
    return result, parent_h, parent_mm, child_h, argmaxi_parent_h, orig, new

def create_down_prop_parallel(
    batch_size: int,
    parent_nodes: int,
    children_per_parent: int,
    parent_dim: int,
    child_dim: int,
    coeff: float = 0.5,
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "down_prop_parallel",
):
    result, parent_h, parent_mm, child_h, argmaxi_parent_h, orig, new = create_tensors(
        batch_size, parent_nodes, children_per_parent, parent_dim, child_dim, coeff, dtype
    )
    
    save_reference(result, dir, f"{name}_result_coeff{coeff}")
    save_reference(parent_h, dir, f"{name}_parent_h_coeff{coeff}")
    save_reference(parent_mm, dir, f"{name}_parent_mm_coeff{coeff}")
    save_reference(child_h, dir, f"{name}_child_h_coeff{coeff}")
    save_reference(argmaxi_parent_h, dir, f"{name}_argmaxi_parent_h_coeff{coeff}")
    save_reference(orig, dir, f"{name}_orig_coeff{coeff}")
    save_reference(new, dir, f"{name}_new_coeff{coeff}")

if __name__ == "__main__":
    create_down_prop_parallel(
        batch_size=6,
        parent_nodes=4,
        children_per_parent=3,
        parent_dim=8,
        child_dim=5,
        coeff=0.5,
        dtype=torch.float32,
        dir="data",
        name="down_prop_parallel"
    )