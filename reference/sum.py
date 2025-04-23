import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable

def create_sum(
    shape: Tuple[int, ...] | List[int] | Iterable[int],
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "sum",
):
    x = torch.randn(shape, dtype=dtype) > 0.1
    y = torch.sum(x, dim=-1, keepdim=True)
    save_reference(x, dir, f"{name}_x")
    save_reference(y, dir, f"{name}_y")

if __name__ == "__main__":
    create_sum((10, 11, 12), dir="data", name="sum_3d")