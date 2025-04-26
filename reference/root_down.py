import torch
import einops
from util.create_layers import LayerInfo
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
from util.hopfield import SparseHopfield

def create_tensors(
    batch_size: int,
    chunk_length: int,
    chunks_per_row: int,
    num_rows: int,
    layers_info: List[LayerInfo],
    dir: str = "data",
    name: str = "root_down",
    eps: float = 1e-6,
):
    assert len(layers_info) == 3, "This test assumes 3 layers"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    net = SparseHopfield(fields, field_dim, layers_info, alpha=16.0, rho=1e-8)
    layer0, layer1, layer2 = net.layers
    sensory_input = torch.rand(batch_size, fields, field_dim)
    upwards = net.up(sensory_input)
    up0, up1, up2 = upwards

    # Inputs to root_down
    root_up = upwards[-1]
    initial_root_counts = net.layer_counts[-1].clone().detach()
    growth_threshold = torch.tensor(net.growth_threshold)

    # Execute root_down
    root_h_sub_l_star = net.root_down(upwards, eps)

    # Get updated counts
    updated_root_counts = net.layer_counts[-1].clone().detach()

    # Save tensors
    save_reference(root_up, dir, f"{name}_root_up")
    save_reference(initial_root_counts, dir, f"{name}_initial_root_counts")
    save_reference(growth_threshold, dir, f"{name}_growth_threshold")
    save_reference(torch.tensor(eps), dir, f"{name}_eps")
    save_reference(root_h_sub_l_star, dir, f"{name}_output")
    save_reference(updated_root_counts, dir, f"{name}_updated_root_counts")

    # Save context tensors (optional but helpful)
    save_reference(layer0, dir, f"{name}_layer0")
    save_reference(layer1, dir, f"{name}_layer1")
    save_reference(layer2, dir, f"{name}_layer2")
    save_reference(sensory_input, dir, f"{name}_sensory_input")
    save_reference(up0, dir, f"{name}_up0")
    save_reference(up1, dir, f"{name}_up1")


if __name__ == "__main__":
    layers_info: List[LayerInfo] = [
        LayerInfo(nodes=-1, memories=16),
        LayerInfo(nodes=4, memories=32),
        LayerInfo(nodes=1, memories=10)
    ]
    # fields is equal to num_rows * chunks_per_row, it must be divisible
    # by the 2nd layer's `nodes` count
    create_tensors(
        batch_size=6,
        chunk_length=7,
        chunks_per_row=4,
        num_rows=2,
        layers_info=layers_info,
        dir="data",
        name="root_down",
        eps=1e-6,
    )
