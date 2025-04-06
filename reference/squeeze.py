import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable


def create_squeeze(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "squeeze",
):
    """
    Create a tensor of random values with the given shape and dtype, add singleton dimensions,
    apply the squeeze operation for each possible dimension, and save the original and squeezed tensors as references.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "squeeze".
    """
    assert len(shape) > 0, "Shape must be a non-empty tuple or list."
    x = torch.randn(shape, dtype=dtype)
    save_reference(x, dir, f"{name}_squeeze_x")  # Save the original tensor

    # Iterate over all possible dimensions and apply squeeze
    for dim in range(x.dim()):
        y = torch.squeeze(x, dim=dim)
        save_reference(y, dir, f"{name}_squeeze_y_dim{dim}")


if __name__ == "__main__":
    d2 = (10, 11)
    d3 = (10, 11, 12)

    create_squeeze(d2, dir="data", name="squeeze2d")
    create_squeeze(d3, dir="data", name="squeeze3d")