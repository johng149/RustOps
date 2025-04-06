import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_sort(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "sort",
):
    """
    Create a tensor and save its sorted values and indices along each dimension.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "sort".
    """
    x = torch.rand(shape, dtype=dtype)
    # Save the original tensor
    save_reference(x, dir, f"{name}_sort_x")
    for dim in range(len(shape)):
        sorted_v, sorted_i = torch.sort(x, dim=dim)
        save_reference(sorted_v, dir, f"{name}_sorted_v_dim{dim}")
        save_reference(sorted_i, dir, f"{name}_sorted_i_dim{dim}")

if __name__ == "__main__":
    create_sort((6, 7, 8, 9, 10, 11, 12), dtype=torch.float32, dir="data", name="sort")