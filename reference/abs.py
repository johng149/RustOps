import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable



def create_abs(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "abs2d",
):
    """
    Create a tensor of random values with the given shape and dtype, compute the absolute values,
    and save the original and absolute tensors as references.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "abs2d".
    """
    x = torch.randn(shape, dtype=dtype)
    y = torch.abs(x)

    save_reference(x, dir, f"{name}_abs_x")
    save_reference(y, dir, f"{name}_abs_y")


if __name__ == "__main__":
    d2 = (10, 11)
    d3 = (10, 11, 12)


    create_abs(d2, dir="data", name="abs2d")
    create_abs(d3, dir="data", name="abs3d")
