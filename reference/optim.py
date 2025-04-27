import torch
import einops
from util.create_layers import LayerInfo
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
from util.hopfield import SparseHopfield

# filepath: /media/john/Tertiary/Projects/ML/RustOps/reference/optim.py

def create_tensors(
    batch_size: int,
    chunk_length: int,
    chunks_per_row: int,
    num_rows: int,
    layers_info: List[LayerInfo],
    dir: str = "data",
    name: str = "optim",
    eps: float = 1e-6,
    coeff: float = 0.5,
):
    assert len(layers_info) == 3, "Expected exactly 3 layers"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    net = SparseHopfield(fields, field_dim, layers_info, t=5, alpha=16.0, rho=1e-8) # Start with t=5

    # Create sensory input
    sensory_input = torch.randn(batch_size, fields, field_dim)

    # Perform up and down passes to get necessary inputs for delta and populate counts
    upwards = net.up(sensory_input)
    # Note: optim is part of the learning step, so we use net.down, not net.pred_down
    downwards = net.down(upwards, eps, coeff) # Returns [root, middle, outer]

    # Calculate deltas needed for optim
    deltas = net.delta(downwards, sensory_input) # Returns [delta_outer, delta_middle, delta_root]

    # Capture state *before* optim
    initial_layer0_weights = net.layers[0].clone().detach()
    initial_layer1_weights = net.layers[1].clone().detach()
    initial_layer2_weights = net.layers[2].clone().detach()
    layer0_counts = net.layer_counts[0].clone().detach()
    layer1_counts = net.layer_counts[1].clone().detach()
    layer2_counts = net.layer_counts[2].clone().detach()
    outer_delta = deltas[0].clone().detach()
    middle_delta = deltas[1].clone().detach()
    root_delta = deltas[2].clone().detach()
    initial_t = torch.tensor(net.t, dtype=torch.int32)

    # Execute optim (modifies net.layers and net.t in place)
    net.optim(deltas)

    # Capture state *after* optim
    final_layer0_weights = net.layers[0].clone().detach()
    final_layer1_weights = net.layers[1].clone().detach()
    final_layer2_weights = net.layers[2].clone().detach()
    final_t = torch.tensor(net.t, dtype=torch.int32)

    # --- Save Tensors ---

    # Inputs to optim
    save_reference(outer_delta, dir, f"{name}_input_delta0")
    save_reference(middle_delta, dir, f"{name}_input_delta1")
    save_reference(root_delta, dir, f"{name}_input_delta2")
    save_reference(initial_layer0_weights, dir, f"{name}_input_layer0_weights_initial")
    save_reference(initial_layer1_weights, dir, f"{name}_input_layer1_weights_initial")
    save_reference(initial_layer2_weights, dir, f"{name}_input_layer2_weights_initial")
    save_reference(layer0_counts, dir, f"{name}_input_layer0_counts")
    save_reference(layer1_counts, dir, f"{name}_input_layer1_counts")
    save_reference(layer2_counts, dir, f"{name}_input_layer2_counts")
    save_reference(initial_t, dir, f"{name}_input_t_initial")
    save_reference(torch.tensor(net.a), dir, f"{name}_input_a")


    # Output of optim
    save_reference(final_layer0_weights, dir, f"{name}_output_layer0_weights_final")
    save_reference(final_layer1_weights, dir, f"{name}_output_layer1_weights_final")
    save_reference(final_layer2_weights, dir, f"{name}_output_layer2_weights_final")
    save_reference(final_t, dir, f"{name}_output_t_final")

    # Context tensors (optional but helpful for debugging/understanding)
    save_reference(sensory_input, dir, f"{name}_context_sensory_input")
    # Save downwards activations as they influence deltas
    save_reference(downwards[0], dir, f"{name}_context_downwards0") # Root
    save_reference(downwards[1], dir, f"{name}_context_downwards1") # Middle
    save_reference(downwards[2], dir, f"{name}_context_downwards2") # Outer


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
        name="optim",
        eps=1e-7,
        coeff=0.6
    )