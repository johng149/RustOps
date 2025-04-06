import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable


def create_scatter(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "scatter2d",
):
    """
    Create a tensor of random values with the given shape and dtype, gather on the last dimension,
    and then scatter the gathered values onto a zeros_like tensor.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "scatter2d".
    """
    x = torch.randn(shape, dtype=dtype)
    indices = torch.argmax(x, dim=-1, keepdim=True)
    gathered = torch.gather(x, -1, indices)

    # Create a zeros_like tensor and scatter the gathered values
    blank = torch.zeros_like(x)
    scattered = torch.scatter(blank, -1, indices, gathered)

    save_reference(x, dir, f"{name}_scatter_x")
    save_reference(indices, dir, f"{name}_scatter_indices")
    save_reference(gathered, dir, f"{name}_scatter_gathered")
    save_reference(blank, dir, f"{name}_scatter_blank")
    save_reference(scattered, dir, f"{name}_scatter_scattered")

if __name__ == "__main__":
    d2 = (10, 11)
    d3 = (10, 11, 12)
    d4 = (10, 11, 12, 13)
    d5 = (10, 11, 12, 13, 14)
    d6 = (10, 11, 12, 13, 14, 15)
    d7 = (10, 11, 12, 13, 14, 15, 16)

    create_scatter(d2, dir="data", name="scatter2d")
    create_scatter(d3, dir="data", name="scatter3d")
    create_scatter(d4, dir="data", name="scatter4d")
    create_scatter(d5, dir="data", name="scatter5d")
    create_scatter(d6, dir="data", name="scatter6d")
    create_scatter(d7, dir="data", name="scatter7d")