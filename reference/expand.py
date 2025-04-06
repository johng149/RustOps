import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_expand(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "expand",
):
    """
    Create a tensor of random values with the given shape and dtype, unsqueeze and expand each dimension by 4,
    and save the original and expanded tensors as references.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "expand".
    """
    assert len(shape) > 0, "Shape must be a non-empty tuple or list."
    x = torch.randn(shape, dtype=dtype)
    save_reference(x, dir, f"{name}_original")  # Save the original tensor

    for dim in range(len(shape) + 1):
        # Unsqueeze and expand the tensor along the current dimension
        y = x.unsqueeze(dim).expand(*x.shape[:dim], 4, *x.shape[dim:])
        save_reference(y, dir, f"{name}_expand_dim_{dim}")

if __name__ == "__main__":
    d2 = (10, 11)
    d3 = (10, 11, 12)

    create_expand(d2, dir="data", name="expand2d")
    create_expand(d3, dir="data", name="expand3d")
