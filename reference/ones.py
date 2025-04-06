from typing import Tuple, List, Iterable
import torch
from util.save_reference import save_reference

def create_ones(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "ones",
):
    """
    Create a tensor of ones with the given shape and dtype.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "../data".
        name (str): Name of the reference tensor file. Default is "ones".
    Returns:
        Tensor: Tensor of ones with the given shape and dtype.
    """
    # Create a tensor of ones
    x = torch.ones(shape, dtype=dtype)
    # Save reference
    save_reference(x, dir, name)
    return x

if __name__ == "__main__":
    create_ones((2, 3), dtype=torch.float32, dir="data", name="ones")