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
    name: str = "delta_outer",
    eps: float = 1e-6,
    coeff: float = 0.5,
):
    assert len(layers_info) == 3, "Expected exactly 3 layers"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    net = SparseHopfield(fields, field_dim, layers_info, alpha=16.0, rho=1e-8)

    layer_weights = [l.clone().detach() for l in net.layers]
    layer_counts = [lc.clone().detach() for lc in net.layer_counts]
    # Create sensory input
    sensory_input = torch.randn(batch_size, fields, field_dim)

    # Perform up and down passes to get necessary inputs for delta_outer
    upwards = net.up(sensory_input)
    # Note: delta_outer is part of the learning step, so we use net.down, not net.pred_down
    downwards = net.down(upwards, eps, coeff)

    # Execute delta_outer
    outer_delta = net.delta_outer(downwards, sensory_input)

    # Save tensors
    save_reference(downwards[0], dir, f"{name}_input_downwards0") # downwards[0] is the outer layer
    save_reference(downwards[1], dir, f"{name}_input_downwards1") # downwards[1] is the root layer
    save_reference(downwards[2], dir, f"{name}_input_downwards2") # downwards[2] is the root layer
    save_reference(net.layers[0], dir, f"{name}_input_layer0") # net.layers[0] is the outer layer
    save_reference(net.layers[1], dir, f"{name}_input_layer1") # net.layers[1] is the root layer
    save_reference(net.layers[2], dir, f"{name}_input_layer2") # net.layers[2] is the root layer
    save_reference(sensory_input, dir, f"{name}_input_sensory")

    # Output
    save_reference(outer_delta, dir, f"{name}_output")

    # Save context tensors (optional but helpful)
    for i, layer_w in enumerate(layer_weights):
        save_reference(layer_w, dir, f"{name}_layer{i}") # Original weights before potential update
    for i, layer_c in enumerate(layer_counts):
         save_reference(layer_c, dir, f"{name}_layer_counts{i}") # Counts after down pass


if __name__ == "__main__":
    layers_info: List[LayerInfo] = [
        LayerInfo(nodes=-1, memories=16), # Outer layer, nodes determined by fields
        LayerInfo(nodes=4, memories=32),
        LayerInfo(nodes=1, memories=10)  # Root layer
    ]
    # fields is equal to num_rows * chunks_per_row, it must be divisible
    # by the 2nd layer's `nodes` count
    create_tensors(
        batch_size=3,
        chunk_length=5,
        chunks_per_row=2,
        num_rows=2, # fields = 4
        layers_info=layers_info,
        dir="data",
        name="delta_outer",
        eps=1e-7,
        coeff=0.6
    )