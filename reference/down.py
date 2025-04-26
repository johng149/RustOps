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
    name: str = "down",
    eps: float = 1e-6,
    coeff: float = 0.5,
):
    assert len(layers_info) == 3, "This test assumes 3 layers"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    net = SparseHopfield(fields, field_dim, layers_info, alpha=16.0, rho=1e-8)
    layer0, layer1, layer2 = net.layers
    sensory_input = torch.rand(batch_size, fields, field_dim)
    upwards = net.up(sensory_input)
    up0, up1, up2 = upwards

    # Inputs to down
    initial_layer_counts = [lc.clone().detach() for lc in net.layer_counts]
    growth_threshold = torch.tensor(net.growth_threshold)

    # Execute down
    downwards = net.down(upwards, eps, coeff)
    down0, down1, down2 = downwards # Note: downwards is reversed order [root, ..., layer0]

    # Get updated counts
    updated_layer_counts = [lc.clone().detach() for lc in net.layer_counts]

    # Save tensors
    save_reference(up0, dir, f"{name}_up0")
    save_reference(up1, dir, f"{name}_up1")
    save_reference(up2, dir, f"{name}_up2")
    save_reference(initial_layer_counts[0], dir, f"{name}_initial_counts0")
    save_reference(initial_layer_counts[1], dir, f"{name}_initial_counts1")
    save_reference(initial_layer_counts[2], dir, f"{name}_initial_counts2")
    save_reference(growth_threshold, dir, f"{name}_growth_threshold")
    save_reference(torch.tensor(eps), dir, f"{name}_eps")
    save_reference(torch.tensor(coeff), dir, f"{name}_coeff")

    # Outputs (downwards is reversed: [root_star, ..., layer0_star])
    save_reference(downwards[0], dir, f"{name}_output_root") # Corresponds to layer 2 output
    save_reference(downwards[1], dir, f"{name}_output_1")   # Corresponds to layer 1 output
    save_reference(downwards[2], dir, f"{name}_output_0")   # Corresponds to layer 0 output

    save_reference(updated_layer_counts[0], dir, f"{name}_updated_counts0")
    save_reference(updated_layer_counts[1], dir, f"{name}_updated_counts1")
    save_reference(updated_layer_counts[2], dir, f"{name}_updated_counts2")

    # Save context tensors (optional but helpful)
    save_reference(layer0, dir, f"{name}_layer0")
    save_reference(layer1, dir, f"{name}_layer1")
    save_reference(layer2, dir, f"{name}_layer2")
    save_reference(sensory_input, dir, f"{name}_sensory_input")


if __name__ == "__main__":
    layers_info: List[LayerInfo] = [
        LayerInfo(nodes=-1, memories=16),
        LayerInfo(nodes=4, memories=32),
        LayerInfo(nodes=1, memories=10)
    ]
    # fields is equal to num_rows * chunks_per_row, it must be divisible
    # by the 2nd layer's `nodes` count
    create_tensors(
        batch_size=2,
        chunk_length=3,
        chunks_per_row=2,
        num_rows=2,
        layers_info=layers_info,
        dir="data",
        name="down",
        eps=1e-6,
        coeff=0.75,
    )