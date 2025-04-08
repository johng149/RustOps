import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_max(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "max",
):
    """
    Create a tensor of max values with the given shape and dtype.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "../data".
        name (str): Name of the reference tensor file. Default is "max".
    """
    x = torch.rand(shape, dtype=dtype)
    # get max_v and max_i for each dimension, and save them separately, also save original `x`
    save_reference(x, dir, f"{name}_max_x")
    for dim in range(len(shape)):
        max_v, max_i = torch.max(x, dim=dim)
        save_reference(max_v, dir, f"{name}_max_v_dim{dim}")
        save_reference(max_i, dir, f"{name}_max_i_dim{dim}")

if __name__ == "__main__":
    create_max((6, 7, 8, 9, 10, 11, 12), dtype=torch.float32, dir="data", name="max")