import torch
from einops import rearrange
from util.save_reference import save_reference
from typing import Tuple, List, Iterable


def test_rearrange(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "rearranged",
):
    """
    Test rearrange operation `batch mems flag -> mems (batch flag)` on a tensor.
    Args:
        shape (Tuple[int, ...]): Shape of the tensor to create.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensor. Default is "data".
        name (str): Name of the reference tensor file. Default is "rearranged".
    """
    reservations = torch.rand(shape, dtype=dtype)
    rearranged = rearrange(reservations, 'batch mems flag -> mems (batch flag)')
    # Save the original tensor
    save_reference(reservations, dir, f"{name}_rearrange_original")
    # Save the rearranged tensor
    save_reference(rearranged, dir, f"{name}_rearrange_result")

if __name__ == "__main__":
    test_rearrange((2, 3, 4), dtype=torch.float32, dir="data", name="rearranged")