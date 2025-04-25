import torch
import einops
from util.save_reference import save_reference
from outer_forward_parallel import create_tensors as create_outer_forward_tensors
from typing import Tuple, List, Iterable

# note that this is pretty much the same as outer_forward_parallel, this is kept here
# to maintain the 1 to 1 mapping with the functions in Rust.
# the python implementation uses a class while the Rust implementation is functional,
# so the interfaces are slightly different but the data used is the same.

def create_tensors(
    batch_size: int,
    fields: int,
    memories: int,
    dim: int,
    rho: float = 1e-8,
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "outer_up",
    mems = None,
    xs = None
):
    # Use the outer_forward_parallel tensors function to get the outer product result
    h_sub_l, mems, xs, numerator, m_norm, x_norm, denom = create_outer_forward_tensors(
        batch_size, fields, memories, dim, rho, dtype, dir, name, mems, xs
    )
    
    # This represents the output of the outer_up function
    # In a real implementation, you might do additional operations here
    result = h_sub_l
    
    return result, mems, xs, h_sub_l

def create_outer_up(
    batch_size: int,
    fields: int,
    memories: int,
    dim: int,
    rho: float = 1e-8,
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "outer_up",
):
    result, mems, xs, h_sub_l = create_tensors(batch_size, fields, memories, dim, rho, dtype, dir, name)
    save_reference(result, dir, f"{name}_result_rho{rho}")
    save_reference(mems, dir, f"{name}_mems_rho{rho}")
    save_reference(xs, dir, f"{name}_xs_rho{rho}")
    save_reference(h_sub_l, dir, f"{name}_h_sub_l_rho{rho}")

if __name__ == "__main__":
    create_outer_up(
        batch_size=6,
        fields=7,
        memories=8,
        dim=5,
        rho=1e-8,
        dtype=torch.float32,
        dir="data",
        name="outer_up"
    )