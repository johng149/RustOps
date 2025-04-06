import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_argmax(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "argmax",
):
    """
    Create a tensor of argmax values with the given shape and dtype.
    Args:
        shape (Tuple[int, ...] | List[int] | Iterable[int]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "../data".
        name (str): Name of the reference tensor file. Default is "argmax".
    """
    x = torch.rand(shape, dtype=dtype)
    # get max_v and max_i for each dimension, and save them separately, also save original `x`
    save_reference(x, dir, f"{name}_argmax_x")
    for dim in range(len(shape)):
        argmaxed = torch.argmax(x, dim=dim)
        argmaxed_keepdim = torch.argmax(x, dim=dim, keepdim=True)
        save_reference(argmaxed, dir, f"{name}argmax_dim{dim}_nokeepdim")
        save_reference(argmaxed_keepdim, dir, f"{name}argmax_dim{dim}_yeskeepdim")

if __name__ == "__main__":
    create_argmax((6, 7, 8, 9, 10, 11), dtype=torch.float32, dir="data", name="argmax")