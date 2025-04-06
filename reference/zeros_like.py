import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable


def create_zeros_like(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "zeros_like",
):
    """
    Create a tensor of zeros with the given shape and dtype, and save it as a reference.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "zeros_like".
    Returns:
        Tensor: Tensor of zeros with the given shape and dtype.
    """
    x = torch.rand(shape, dtype=dtype)
    zeros = torch.zeros_like(x)
    # save zeros and original `x`
    save_reference(zeros, dir, f"{name}_zeros_like")
    save_reference(x, dir, f"{name}_zeros_like_x")

if __name__ == "__main__":
    create_zeros_like((6, 7, 8, 9, 10, 11, 12), dtype=torch.float32, dir="data", name="zeros_like")