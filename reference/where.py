import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_where(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    threshold: float = 0.5,
    dir: str = "data",
    name: str = "where",
):
    """
    Create a tensor and save the results of a where operation based on a threshold.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        threshold (float): Threshold value for the where condition. Default is 0.5.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "where".
    """
    x = torch.rand(shape, dtype=dtype)
    # Save the original tensor
    save_reference(x, dir, f"{name}_where_x")
    # Apply the where condition
    condition = x > threshold
    result = torch.where(condition, x, torch.zeros_like(x))
    # Save the condition and result tensors
    save_reference(condition, dir, f"{name}_condition")
    save_reference(result, dir, f"{name}_result")

if __name__ == "__main__":
    create_where((11, 12), dtype=torch.float32, threshold=0.5, dir="data", name="where")