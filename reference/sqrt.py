import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

# filepath: /media/john/Tertiary/Projects/ML/RustOps/reference/sqrt.py


def create_sqrt(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "sqrt2d",
):
    """
    Create a tensor of random non-negative values with the given shape and dtype, compute the square roots,
    and save the original and square root tensors as references.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "sqrt2d".
    """
    x = torch.rand(shape, dtype=dtype)  # Generate random values in [0, 1)
    y = torch.sqrt(x)

    save_reference(x, dir, f"{name}_sqrt_x")
    save_reference(y, dir, f"{name}_sqrt_y")


if __name__ == "__main__":
    d2 = (10, 11)
    d3 = (10, 11, 12)

    create_sqrt(d2, dir="data", name="sqrt2d")
    create_sqrt(d3, dir="data", name="sqrt3d")