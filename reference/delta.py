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
    name: str = "delta",
    eps: float = 1e-6,
    coeff: float = 0.5,
):
    assert len(layers_info) == 3, "Expected exactly 3 layers"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    net = SparseHopfield(fields, field_dim, layers_info, alpha=16.0, rho=1e-8)

    layer_weights = [l.clone().detach() for l in net.layers]

    # Create sensory input
    sensory_input = torch.randn(batch_size, fields, field_dim)

    # Perform up and down passes to get necessary inputs for delta
    upwards = net.up(sensory_input)
    # Note: delta is part of the learning step, so we use net.down, not net.pred_down
    downwards = net.down(upwards, eps, coeff) # Returns [root, middle, outer]
    layer_counts = [lc.clone().detach() for lc in net.layer_counts] # Capture counts after down pass

    # Execute delta
    deltas = net.delta(downwards, sensory_input) # Returns [delta_outer, delta_middle, delta_root]

    # Save inputs
    save_reference(downwards[0], dir, f"{name}_input_downwards0") # Outer layer activations
    save_reference(downwards[1], dir, f"{name}_input_downwards1") # Middle layer activations
    save_reference(downwards[2], dir, f"{name}_input_downwards2") # Root layer activations

    save_reference(sensory_input, dir, f"{name}_input_sensory")

    save_reference(net.layers[0], dir, f"{name}_input_layer0") # Outer layer weights
    save_reference(net.layers[1], dir, f"{name}_input_layer1") # Middle layer weights
    save_reference(net.layers[2], dir, f"{name}_input_layer2") # Root layer weights

    # Save outputs
    save_reference(deltas[0], dir, f"{name}_output_delta0") # Delta for outer layer
    save_reference(deltas[1], dir, f"{name}_output_delta1") # Delta for middle layer
    save_reference(deltas[2], dir, f"{name}_output_delta2") # Delta for root layer

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
        name="delta",
        eps=1e-7,
        coeff=0.6
    )