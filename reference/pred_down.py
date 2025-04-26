import torch
import einops
from util.create_layers import LayerInfo
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
from util.hopfield import SparseHopfield

# filepath: /media/john/Tertiary/Projects/ML/RustOps/reference/pred_down.py

def create_tensors(
    batch_size: int,
    chunk_length: int,
    chunks_per_row: int,
    num_rows: int,
    layers_info: List[LayerInfo],
    dir: str = "data",
    name: str = "pred_down",
    eps: float = 1e-6,
    coeff: float = 0.5,
):
    assert len(layers_info) >= 1, "Need at least one layer"
    assert len(layers_info) == 3, "Need exactly three layers"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    net = SparseHopfield(fields, field_dim, layers_info, alpha=16.0, rho=1e-8)

    layer_weights = [l.clone().detach() for l in net.layers]
    layer_counts = [lc.clone().detach() for lc in net.layer_counts] # Save for context

    # Create sensory input
    sensory_input = torch.randn(batch_size, fields, field_dim)
    upwards = net.up(sensory_input)

    # Execute pred_down
    downwards = net.pred_down(upwards, eps, coeff)

    # Save tensors
    # Inputs
    save_reference(upwards[0], dir, f"{name}_input_up0")
    save_reference(upwards[1], dir, f"{name}_input_up1")
    save_reference(upwards[2], dir, f"{name}_input_up2")
    save_reference(torch.tensor(eps), dir, f"{name}_eps")
    save_reference(torch.tensor(coeff), dir, f"{name}_coeff")
    save_reference(layer_weights[0], dir, f"{name}_layer0") # Layer weights are needed for down_prop_parallel
    save_reference(layer_weights[1], dir, f"{name}_layer1")
    save_reference(layer_weights[2], dir, f"{name}_layer2")

    # Output
    save_reference(downwards[0], dir, f"{name}_output_down0")
    save_reference(downwards[1], dir, f"{name}_output_down1")
    save_reference(downwards[2], dir, f"{name}_output_down2")

    # Save context tensors (optional but helpful)
    save_reference(layer_counts[0], dir, f"{name}_layer_counts0") # Context, not directly used but part of net state
    save_reference(layer_counts[1], dir, f"{name}_layer_counts1")
    save_reference(layer_counts[2], dir, f"{name}_layer_counts2")
    save_reference(sensory_input, dir, f"{name}_sensory_input") # Context


if __name__ == "__main__":
    layers_info: List[LayerInfo] = [
        LayerInfo(nodes=-1, memories=3), # Outer layer, nodes will be set to fields
        LayerInfo(nodes=4, memories=3),
        LayerInfo(nodes=1, memories=3)  # Root layer
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
        name="pred_down",
        eps=1e-7,
        coeff=0.3,
    )