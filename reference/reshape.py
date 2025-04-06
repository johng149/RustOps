import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

# filepath: /media/john/Tertiary/Projects/ML/RustOps/reference/reshape.py

def create_reshape(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    new_shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "reshape",
):
    """
    Create a tensor, reshape it to a 2D shape, and save both the original and reshaped tensors.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        new_shape (Tuple[int, int]): New 2D shape to reshape the tensor into.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "reshape".
    """
    x = torch.rand(shape, dtype=dtype)
    # Save the original tensor
    save_reference(x, dir, f"{name}_reshape_original")
    # Reshape the tensor
    reshaped_x = x.reshape(new_shape)
    save_reference(reshaped_x, dir, f"{name}_reshape_reshaped")

if __name__ == "__main__":
    create_reshape((8, 3, 10), new_shape=(8, -1), dtype=torch.float32, dir="data", name="reshape2d")
    create_reshape((3, 8, 10), new_shape=(3, 1, -1), dtype=torch.float32, dir="data", name="reshape3d")
    create_reshape((3, 8, 10), new_shape=(3, 4, 2, 10), dtype=torch.float32, dir="data", name="reshape4d")