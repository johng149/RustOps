import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
import einops

def create_tensors(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        batch_size: int = 6,
        dtype: torch.dtype = torch.long,
        dir: str = "data",
        name: str = "expand_for_batches",
        x = None,
):
    assert (shape is None and x is not None) or len(shape) == 2, "Shape must be 2D"
    x = torch.randint(0, 4, shape, dtype=dtype) if x is None else x
    nodes, mems = x.shape
    full_expands = (batch_size // mems) + 1
    result = x.unsqueeze(0).expand(full_expands, -1, -1).transpose(0, 1).reshape(nodes, -1)[:, :batch_size]
    return result, x

def create_expand_for_batches(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        batch_size: int = 6,
        dtype: torch.dtype = torch.long,
        dir: str = "data",
        name: str = "expand_for_batches",
):
    result, x = create_tensors(shape, batch_size, dtype, dir, name)
    save_reference(result, dir, f"{name}_{batch_size}_result")
    save_reference(x, dir, f"{name}_{batch_size}_x")

if __name__ == "__main__":
    # batch size < mems
    create_expand_for_batches((6, 7), batch_size=4, dtype=torch.long, dir="data", name="expand_for_batches")
    # batch size = mems
    create_expand_for_batches((6, 7), batch_size=7, dtype=torch.long, dir="data", name="expand_for_batches")
    # batch size > mems, divisible
    create_expand_for_batches((6, 7), batch_size=14, dtype=torch.long, dir="data", name="expand_for_batches")
    # batch size > mems, not divisible
    create_expand_for_batches((6, 7), batch_size=15, dtype=torch.long, dir="data", name="expand_for_batches")