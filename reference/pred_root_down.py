import torch
import einops
from util.create_layers import LayerInfo
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
from util.hopfield import SparseHopfield

# filepath: /media/john/Tertiary/Projects/ML/RustOps/reference/pred_root_down.py

def create_tensors(
    batch_size: int,
    chunk_length: int,
    chunks_per_row: int,
    num_rows: int,
    layers_info: List[LayerInfo],
    dir: str = "data",
    name: str = "pred_root_down",
    eps: float = 1e-6,
):
    assert len(layers_info) >= 1, "Need at least one layer"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    net = SparseHopfield(fields, field_dim, layers_info, alpha=16.0, rho=1e-8)

    layer_weights = [l.clone().detach() for l in net.layers]
    layer_counts = [lc.clone().detach() for lc in net.layer_counts] # Save for context if needed

    # Create sensory input
    sensory_input = torch.randn(batch_size, fields, field_dim)
    upwards = net.up(sensory_input)

    # Execute pred_root_down
    root_h_sub_l_star = net.pred_root_down(upwards, eps)

    # Save tensors
    # Inputs
    save_reference(upwards[0], dir, f"{name}_input_up0") # Save each upwards element explicitly
    save_reference(upwards[1], dir, f"{name}_input_up1")
    save_reference(upwards[2], dir, f"{name}_input_up2") # This is also root_h_sub_l
    save_reference(torch.tensor(eps), dir, f"{name}_eps")

    # Output
    save_reference(root_h_sub_l_star, dir, f"{name}_output")

    # Save context tensors (optional but helpful)
    # Removed saving upwards elements here as they are now inputs
    for i, layer_w in enumerate(layer_weights):
        save_reference(layer_w, dir, f"{name}_layer{i}")
    for i, layer_c in enumerate(layer_counts):
         save_reference(layer_c, dir, f"{name}_layer_counts{i}") # Context
    # Keep sensory_input for context if needed, though not directly used in pred_root_down
    # save_reference(sensory_input, dir, f"{name}_sensory_input") # Removed as sensory_input is not available here



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
        num_rows=2, # fields = 4
        layers_info=layers_info,
        dir="data",
        name="pred_root_down",
        eps=1e-7,
    )