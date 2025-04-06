import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable


def create_gather(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "gather2d",
):
    """
    Create a tensor of random values with the given shape and dtype, and then call gather on the
    last dimension
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "gather2d".
    """
    x = torch.randn(shape, dtype=dtype)
    indices = torch.argmax(x, dim=-1, keepdim=True)
    y = torch.gather(x, dim=-1, index=indices)

    save_reference(x, dir, f"{name}_gather_x")
    save_reference(indices, dir, f"{name}_gather_indices")
    save_reference(y, dir, f"{name}_gather_y")

if __name__ == "__main__":
    d2 = (10, 11)
    d3 = (10, 11, 12)
    d4 = (10, 11, 12, 13)
    d5 = (10, 11, 12, 13, 14)
    d6 = (10, 11, 12, 13, 14, 15)
    d7 = (10, 11, 12, 13, 14, 15, 16)

    create_gather(d2, dir="data", name="gather2d")
    create_gather(d3, dir="data", name="gather3d")
    create_gather(d4, dir="data", name="gather4d")
    create_gather(d5, dir="data", name="gather5d")
    create_gather(d6, dir="data", name="gather6d")
    create_gather(d7, dir="data", name="gather7d")