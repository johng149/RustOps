import torch
import einops
from util.create_layers import LayerInfo
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
from util.hopfield import SparseHopfield

def create_incremental_optimize_tensors(
    batch_size: int,
    chunk_length: int,
    chunks_per_row: int,
    num_rows: int,
    layers_info: List[LayerInfo],
    dir: str = "data",
    name: str = "incremental_optimize",
    eps: float = 1e-6,
    coeff: float = 0.5,
    alpha: float = 1.0,
    rho: float = 1e-8,
    t: int = 0,
):
    """
    Generates input and output tensors for testing the SparseHopfield.optimize method
    in an incremental fashion. It first optimizes on the first sample, then on the rest.
    Requires exactly 3 layers.
    """
    assert len(layers_info) == 3, "Expected exactly 3 layers (outer, middle, root)"
    assert batch_size > 1, "Batch size must be greater than 1 for incremental test"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    net = SparseHopfield(fields, field_dim, layers_info, t=t, alpha=alpha, rho=rho)

    # Create sensory input
    sensory_input = torch.randn(batch_size, fields, field_dim)

    # Capture initial state
    initial_layer0 = net.layers[0].clone().detach()
    initial_layer1 = net.layers[1].clone().detach()
    initial_layer2 = net.layers[2].clone().detach()
    initial_count0 = net.layer_counts[0].clone().detach()
    initial_count1 = net.layer_counts[1].clone().detach()
    initial_count2 = net.layer_counts[2].clone().detach()
    initial_t = torch.tensor(net.t, dtype=torch.int32)
    initial_growth_threshold = torch.tensor(net.growth_threshold, dtype=torch.float32)

    # --- Save Inputs ---
    save_reference(sensory_input, dir, f"{name}_input_sensory_input")
    save_reference(initial_layer0, dir, f"{name}_input_layer0_weights_initial")
    save_reference(initial_layer1, dir, f"{name}_input_layer1_weights_initial")
    save_reference(initial_layer2, dir, f"{name}_input_layer2_weights_initial")
    save_reference(initial_count0, dir, f"{name}_input_layer0_counts_initial")
    save_reference(initial_count1, dir, f"{name}_input_layer1_counts_initial")
    save_reference(initial_count2, dir, f"{name}_input_layer2_counts_initial")
    save_reference(initial_t, dir, f"{name}_input_t_initial")
    save_reference(torch.tensor(eps), dir, f"{name}_input_eps")
    save_reference(torch.tensor(coeff), dir, f"{name}_input_coeff")
    save_reference(torch.tensor(alpha), dir, f"{name}_input_alpha")
    save_reference(torch.tensor(rho), dir, f"{name}_input_rho")
    save_reference(initial_growth_threshold, dir, f"{name}_input_growth_threshold")


    # --- Execute the incremental optimize method ---
    # This modifies the network's layers and counts in place
    # First, optimize on the first element
    print("Optimizing on the first sample")
    net.optimize(sensory_input[0:1], eps, coeff)
    # Then, optimize on the rest of the batch
    print("Optimizing on the rest of the batch")
    net.optimize(sensory_input[1:], eps, coeff)

    # --- Capture Final State ---
    final_layer0 = net.layers[0].clone().detach()
    final_layer1 = net.layers[1].clone().detach()
    final_layer2 = net.layers[2].clone().detach()
    final_count0 = net.layer_counts[0].clone().detach()
    final_count1 = net.layer_counts[1].clone().detach()
    final_count2 = net.layer_counts[2].clone().detach()
    final_t = torch.tensor(net.t, dtype=torch.int32)
    final_growth_threshold = torch.tensor(net.growth_threshold, dtype=torch.float32)

    # --- Save Outputs ---
    save_reference(final_layer0, dir, f"{name}_output_layer0_weights_final")
    save_reference(final_layer1, dir, f"{name}_output_layer1_weights_final")
    save_reference(final_layer2, dir, f"{name}_output_layer2_weights_final")
    save_reference(final_count0, dir, f"{name}_output_layer0_counts_final")
    save_reference(final_count1, dir, f"{name}_output_layer1_counts_final")
    save_reference(final_count2, dir, f"{name}_output_layer2_counts_final")
    save_reference(final_t, dir, f"{name}_output_t_final")
    save_reference(final_growth_threshold, dir, f"{name}_output_growth_threshold_final")

if __name__ == "__main__":
    layers_info_3: List[LayerInfo] = [
        LayerInfo(nodes=-1, memories=4), # Outer layer, nodes determined by fields
        LayerInfo(nodes=2, memories=4),
        LayerInfo(nodes=1, memories=4)  # Root layer
    ]
    # fields = num_rows * chunks_per_row = 16 * 16 = 256
    # Layer 1 nodes (4) must divide fields (256) evenly. 256 % 4 == 0. OK.
    # Layer 2 nodes (1) must divide Layer 1 nodes (4) evenly. 4 % 1 == 0. OK.
    create_incremental_optimize_tensors(
        batch_size=2,
        chunk_length=4,
        chunks_per_row=4,
        num_rows=4,
        layers_info=layers_info_3,
        dir="data",
        name="incremental_optimize_3layer",
        eps=1e-7,
        coeff=0.6,
        alpha=16.0,
        rho=1e-9,
        t=5
    )