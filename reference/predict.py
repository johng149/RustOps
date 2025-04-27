import torch
import einops
from util.create_layers import LayerInfo
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
from util.hopfield import SparseHopfield

# filepath: /media/john/Tertiary/Projects/ML/RustOps/reference/predict.py

def create_predict_tensors(
    batch_size: int,
    chunk_length: int,
    chunks_per_row: int,
    num_rows: int,
    layers_info: List[LayerInfo],
    dir: str = "data",
    name: str = "predict",
    eps: float = 1e-6,
    coeff: float = 0.5,
    alpha: float = 1.0, # Not directly used by predict, but part of net state
    rho: float = 1e-8,  # Used in up pass
    t: int = 0,         # Not directly used by predict, but part of net state
):
    """
    Generates input and output tensors for testing the SparseHopfield.predict method.
    Requires exactly 3 layers.
    """
    assert len(layers_info) == 3, "Expected exactly 3 layers (outer, middle, root)"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    # Initialize network with some potentially trained state
    # For reproducibility, we'll use fixed random state for initialization here
    # In a real scenario, these weights/counts would come from training
    torch.manual_seed(42)
    net = SparseHopfield(fields, field_dim, layers_info, t=t, alpha=alpha, rho=rho)
    # Simulate some training steps to get non-zero weights/counts
    for _ in range(t + 1): # Simulate t+1 steps to match the initial t value
        dummy_input = torch.rand(batch_size, fields, field_dim)
        net.optimize(dummy_input, eps=eps, coeff=coeff)
    net.reset_iteration(t) # Reset t back to the desired initial value for the test

    # Create sensory input for prediction
    sensory_input = torch.randn(batch_size, fields, field_dim)

    # Capture network state (needed as input for prediction)
    layer0_weights = net.layers[0].clone().detach()
    layer1_weights = net.layers[1].clone().detach()
    layer2_weights = net.layers[2].clone().detach()
    # Counts are not strictly needed for predict, but save for completeness
    layer0_counts = net.layer_counts[0].clone().detach()
    layer1_counts = net.layer_counts[1].clone().detach()
    layer2_counts = net.layer_counts[2].clone().detach()

    # --- Save Inputs ---
    save_reference(sensory_input, dir, f"{name}_input_sensory_input")
    save_reference(layer0_weights, dir, f"{name}_input_layer0_weights")
    save_reference(layer1_weights, dir, f"{name}_input_layer1_weights")
    save_reference(layer2_weights, dir, f"{name}_input_layer2_weights")
    save_reference(layer0_counts, dir, f"{name}_input_layer0_counts") # For completeness
    save_reference(layer1_counts, dir, f"{name}_input_layer1_counts") # For completeness
    save_reference(layer2_counts, dir, f"{name}_input_layer2_counts") # For completeness
    save_reference(torch.tensor(eps), dir, f"{name}_input_eps")
    save_reference(torch.tensor(coeff), dir, f"{name}_input_coeff")
    save_reference(torch.tensor(rho), dir, f"{name}_input_rho") # Needed for up pass
    # Save alpha and t for potential state reconstruction, though not used by predict directly
    save_reference(torch.tensor(alpha), dir, f"{name}_input_alpha")
    save_reference(torch.tensor(t, dtype=torch.int32), dir, f"{name}_input_t")


    # --- Execute the predict method ---
    prediction = net.predict(sensory_input, eps, coeff)

    # --- Save Outputs ---
    save_reference(prediction, dir, f"{name}_output_prediction")


if __name__ == "__main__":
    layers_info_3: List[LayerInfo] = [
        LayerInfo(nodes=-1, memories=16), # Outer layer, nodes determined by fields
        LayerInfo(nodes=4, memories=32),
        LayerInfo(nodes=1, memories=16)  # Root layer
    ]
    # fields = num_rows * chunks_per_row = 16 * 16 = 256
    # Layer 1 nodes (4) must divide fields (256) evenly. 256 % 4 == 0. OK.
    # Layer 2 nodes (1) must divide Layer 1 nodes (4) evenly. 4 % 1 == 0. OK.
    create_predict_tensors(
        batch_size=10,
        chunk_length=32,
        chunks_per_row=16,
        num_rows=16, # fields = 256
        layers_info=layers_info_3,
        dir="data",
        name="predict_3layer",
        eps=1e-7,
        coeff=0.6,
        alpha=16.0,
        rho=1e-9,
        t=5 # Simulate a network that has seen 5 updates
    )