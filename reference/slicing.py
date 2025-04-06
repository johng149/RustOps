import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable


def test_slicingccn1(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "sliced",
):
    """
    Test slicing operation `[:, :, -1:]` on a tensor.
    Args:
        shape (Tuple[int, ...]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "sliced".
    """
    x = torch.rand(shape, dtype=dtype)
    y = x[:, :, -1:]
    # Save the original tensor
    save_reference(x, dir, f"{name}_sliced[:, :, -1:]_x")
    # Save the sliced tensor
    save_reference(y, dir, f"{name}_sliced[:, :, -1:]_y")

def test_slicingcb(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "sliced",
):
    """
    Test slicing operation `[:, :batch_size]` on a tensor. Here `batch_size` is the second dimension of the tensor
    minus one
    Args:
        shape (Tuple[int, ...]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "sliced".
    """
    x = torch.rand(shape, dtype=dtype)
    y = x[:, :shape[1]-1]
    # Save the original tensor
    save_reference(x, dir, f"{name}_sliced[:, :batch_size]_x")
    # Save the sliced tensor
    save_reference(y, dir, f"{name}_sliced[:, :batch_size]_y")


if __name__ == "__main__":
    test_slicingccn1((2, 3, 4), dtype=torch.float32, dir="data", name="sliced")
    test_slicingcb((3, 4), dtype=torch.float32, dir="data", name="sliced")