import torch
import einops
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_tensors(
    batch_size: int,
    fields: int,
    memories: int,
    dim: int,
    rho: float = 1e-8,
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "outer_forward_parallel",
    mems = None,
    xs = None
):
    # Create random tensors if not provided
    if mems is None:
        mems = torch.rand((batch_size, fields, memories, dim), dtype=dtype)
    elif mems is not None and mems.dim() == 3:
        mems = mems.expand(batch_size, -1, -1, -1)
    if xs is None:
        xs = torch.rand((batch_size, fields, dim), dtype=dtype)
    
    # Execute the outer_forward_parallel logic
    m = mems - 0.5
    x = xs - 0.5
    numerator = einops.einsum(m, x, 'batch fields memories dim, batch fields dim -> batch fields memories') * 0.5
    m_norm = torch.sqrt(einops.reduce(m ** 2, 'batch fields memories dim -> batch fields memories', 'sum'))
    x_norm = torch.sqrt(einops.reduce(x ** 2, 'batch fields dim -> batch fields', 'sum')).unsqueeze(-1)
    denom = m_norm * x_norm + rho
    result = (numerator / denom) + 0.5
    
    return result, mems, xs, numerator, m_norm, x_norm, denom

def create_outer_forward_parallel(
    batch_size: int,
    fields: int,
    memories: int,
    dim: int,
    rho: float = 1e-8,
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "outer_forward_parallel",
):
    result, mems, xs, numerator, m_norm, x_norm, denom = create_tensors(batch_size, fields, memories, dim, rho, dtype, dir, name)
    save_reference(result, dir, f"{name}_result_rho{rho}")
    save_reference(mems, dir, f"{name}_mems_rho{rho}")
    save_reference(xs, dir, f"{name}_xs_rho{rho}")
    save_reference(numerator, dir, f"{name}_numerator_rho{rho}")
    save_reference(m_norm, dir, f"{name}_m_norm_rho{rho}")
    save_reference(x_norm, dir, f"{name}_x_norm_rho{rho}")
    save_reference(denom, dir, f"{name}_denom_rho{rho}")

if __name__ == "__main__":
    create_outer_forward_parallel(
    batch_size=6, 
    fields=7, 
    memories=8, 
    dim=5, 
    rho=1e-8, 
    dtype=torch.float32,
    dir="data", 
    name="outer_forward_parallel"
    )