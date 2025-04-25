import torch
import einops
from util.create_layers import LayerInfo
from util.create_layers import create_layers
from util.save_reference import save_reference
from outer_forward_parallel import create_tensors as create_outer_forward_tensors
from typing import Tuple, List, Iterable

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
    layers_info: List[LayerInfo] = [
        LayerInfo(nodes=-1, memories=16),
        LayerInfo(nodes=3, memories=32),
        LayerInfo(nodes=1, memories=10)
    ]
    layers, layer_counts = create_layers(layers_info, fields, dim)
    result, mems, xs, h_sub_l = create_tensors(batch_size, fields, memories, dim, rho, dtype, dir, name, mems=layers[0])
    save_reference(result, dir, f"{name}_result_rho{rho}")
    save_reference(mems, dir, f"{name}_mems_rho{rho}")
    save_reference(xs, dir, f"{name}_xs_rho{rho}")
    save_reference(h_sub_l, dir, f"{name}_h_sub_l_rho{rho}")

if __name__ == "__main__":
    create_outer_up(
        batch_size=6,
        fields=9,
        memories=8,
        dim=5,
        rho=1e-8,
        dtype=torch.float32,
        dir="data",
        name="outer_up"
    )