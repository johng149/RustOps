import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_transpose(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "transpose",
):
    """
    Create a tensor and save the results of transposing every adjacent pair of dimensions.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "transpose".
    """
    x = torch.rand(shape, dtype=dtype)
    # Save the original tensor
    save_reference(x, dir, f"{name}_transpose_x")
    # Apply transpose for every adjacent pair of dimensions
    for i in range(len(shape) - 1):
        transposed = x.transpose(i, i + 1)
        # Save each transposed tensor
        save_reference(transposed, dir, f"{name}_transpose_{i}_{i+1}")

if __name__ == "__main__":
    create_transpose((11, 12, 13), dtype=torch.float32, dir="data", name="transpose")