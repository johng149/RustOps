import torch
from einops import reduce
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def test_reduce_sum(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "reduced",
):
    """
    Test reduction operation `einops.reduce(grown, 'batch nodes mems -> nodes mems', 'sum')` on a tensor.
    Args:
        shape (Tuple[int, ...]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "reduced".
    """
    x = torch.rand(shape, dtype=dtype)
    y = reduce(x, 'batch nodes mems -> nodes mems', 'sum')
    # Save the original tensor
    save_reference(x, dir, f"{name}_reduce_bnm_nm_reduce_sum_x")
    # Save the reduced tensor
    save_reference(y, dir, f"{name}_reduce_bnm_nm_reduce_sum_y")


def test_reduce_sum_batch_fields_memories(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "reduced_batch_fields_memories",
):
    """
    Test reduction operation `einops.reduce(x, 'batch fields memories dim -> batch fields memories', 'sum')` on a tensor.
    Args:
        shape (Tuple[int, ...]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "reduced_batch_fields_memories".
    """
    x = torch.rand(shape, dtype=dtype)
    y = reduce(x, 'batch fields memories dim -> batch fields memories', 'sum')
    # Save the original tensor
    save_reference(x, dir, f"{name}_reduce_bfmd_bfm_x")
    # Save the reduced tensor
    save_reference(y, dir, f"{name}_reduce_bfmd_bfm_y")


def test_reduce_sum_batch_field(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "reduced_batch_field",
):
    """
    Test reduction operation `einops.reduce(x, 'batch field dim -> batch field', 'sum')` on a tensor.
    Args:
        shape (Tuple[int, ...]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "reduced_batch_field".
    """
    x = torch.rand(shape, dtype=dtype)
    y = reduce(x, 'batch field dim -> batch field', 'sum')
    # Save the original tensor
    save_reference(x, dir, f"{name}_reduce_bfd_bf_x")
    # Save the reduced tensor
    save_reference(y, dir, f"{name}_reduce_bfd_bf_y")

def test_reduce_sum_batch_hidden(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "reduced_batch_hidden",
):
    """
    Test reduction operation `einops.reduce(x, 'batch hidden children c_mems -> batch hidden', 'sum')` on a tensor.
    Args:
        shape (Tuple[int, ...]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "reduced_batch_hidden".
    """
    x = torch.rand(shape, dtype=dtype)
    y = reduce(x, 'batch hidden children c_mems -> batch hidden', 'sum')
    # Save the original tensor
    save_reference(x, dir, f"{name}_reduce_bhccm_bh_x")
    # Save the reduced tensor
    save_reference(y, dir, f"{name}_reduce_bhccm_bh_y")


if __name__ == "__main__":
    test_reduce_sum((2, 3, 4), dtype=torch.float32, dir="data", name="reduced")
    test_reduce_sum_batch_fields_memories((2, 3, 4, 5), dtype=torch.float32, dir="data", name="reduced_batch_fields_memories")
    test_reduce_sum_batch_field((2, 3, 4), dtype=torch.float32, dir="data", name="reduced_batch_field")
    test_reduce_sum_batch_hidden((2, 3, 5, 6), dtype=torch.float32, dir="data", name="reduced_batch_hidden")