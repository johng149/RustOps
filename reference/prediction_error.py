import torch
import einops
from util.create_layers import LayerInfo
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
from util.hopfield import SparseHopfield

# filepath: /media/john/Tertiary/Projects/ML/RustOps/reference/prediction_error.py

def create_prediction_error_tensors(
    batch_size: int,
    chunk_length: int,
    chunks_per_row: int,
    num_rows: int,
    layers_info: List[LayerInfo],
    dir: str = "data",
    name: str = "prediction_error",
    eps: float = 1e-6,
    coeff: float = 0.5,
    alpha: float = 1.0, # Defines growth threshold, affects training
    rho: float = 1e-8,  # Used in up pass
    t: int = 0,         # Number of simulated training steps
):
    """
    Generates input and output tensors for testing the SparseHopfield.prediction_error method.
    Requires exactly 3 layers.
    The 't' parameter determines if the network state reflects training (t>0) or not (t=0).
    If t=1, the network is trained exactly once on the generated sensory_input.
    If t>1, the network is trained t times on dummy data.
    """
    assert len(layers_info) == 3, "Expected exactly 3 layers (outer, middle, root)"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    # Initialize network
    # Use fixed seed for reproducibility of initial weights
    torch.manual_seed(42)
    net = SparseHopfield(fields, field_dim, layers_info, t=0, alpha=alpha, rho=rho) # Start t at 0

    # Create sensory input for prediction error calculation AND potential single training step
    # Use a different seed for the actual test input
    torch.manual_seed(123)
    sensory_input = torch.randn(batch_size, fields, field_dim)

    # Simulate training steps if t > 0
    if t > 0:
        if t == 1:
            # Train exactly once on the sensory_input that will be used for the test
            net.optimize(sensory_input, eps=eps, coeff=coeff)
        else:
            # Simulate multiple training steps using consistent dummy data
            for i in range(t):
                dummy_input = torch.rand(batch_size, fields, field_dim, generator=torch.Generator().manual_seed(42 + i))
                net.optimize(dummy_input, eps=eps, coeff=coeff)
        # Ensure the network's internal t matches the specified t for the test state
        assert net.t == t
    else:
        # If t=0, ensure network t is 0
        assert net.t == 0

    # Capture network state (needed as input for prediction_error)
    layer0_weights = net.layers[0].clone().detach()
    layer1_weights = net.layers[1].clone().detach()
    layer2_weights = net.layers[2].clone().detach()
    # Counts are not strictly needed for prediction_error, but save for completeness
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
    save_reference(torch.tensor(rho), dir, f"{name}_input_rho") # Needed for internal predict -> up pass
    # Save alpha and t to define the network state used
    save_reference(torch.tensor(alpha), dir, f"{name}_input_alpha")
    save_reference(torch.tensor(t, dtype=torch.int32), dir, f"{name}_input_t")


    # --- Execute the prediction_error method ---
    # Ensure no_grad context as prediction_error itself uses it
    with torch.no_grad():
        error = net.prediction_error(sensory_input, eps, coeff)

    # --- Save Outputs ---
    save_reference(error, dir, f"{name}_output_error")


if __name__ == "__main__":
    layers_info_3: List[LayerInfo] = [
        LayerInfo(nodes=-1, memories=8), # Outer layer, nodes determined by fields
        LayerInfo(nodes=2, memories=16),
        LayerInfo(nodes=1, memories=8)  # Root layer
    ]
    # fields = num_rows * chunks_per_row = 4 * 4 = 16
    # Layer 1 nodes (2) must divide fields (16) evenly. 16 % 2 == 0. OK.
    # Layer 2 nodes (1) must divide Layer 1 nodes (2) evenly. 2 % 1 == 0. OK.

    common_params = {
        "batch_size": 5,
        "chunk_length": 16,
        "chunks_per_row": 4,
        "num_rows": 4, # fields = 16
        "layers_info": layers_info_3,
        "dir": "data",
        "eps": 1e-7,
        "coeff": 0.4,
        "alpha": 10.0,
        "rho": 1e-9,
    }

    # Untrained network test
    create_prediction_error_tensors(
        **common_params,
        name="prediction_error_untrained",
        t=0 # Untrained state
    )

    # Trained network test (trained once on the test input)
    create_prediction_error_tensors(
        **common_params,
        name="prediction_error_trained",
        t=1 # Simulate a network that has seen exactly 1 update on the test data
    )
