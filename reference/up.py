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
    name: str = "up",
):
    assert len(layers_info) == 3, "This test assumes 3 layers"
    fields = num_rows * chunks_per_row
    field_dim = chunk_length

    net = SparseHopfield(fields, field_dim, layers_info, alpha=16.0, rho=1e-8)
    layer0, layer1, layer2 = net.layers
    sensory_input = torch.rand(batch_size, fields, field_dim)
    uped = net.up(sensory_input)

    up0, up1, up2 = uped
    save_reference(up0, dir, f"{name}_up0")
    save_reference(up1, dir, f"{name}_up1")
    save_reference(up2, dir, f"{name}_up2")
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
        batch_size=6,
        chunk_length=7,
        chunks_per_row=4,
        num_rows=2,
        layers_info=layers_info,
        dir="data",
        name="up",
    )