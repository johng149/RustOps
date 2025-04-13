import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable


def create_tensors(shape: Tuple[int, ...] | List[int] | Iterable[int], dtype: torch.dtype = torch.float32):
    x = torch.rand(shape, dtype=dtype)
    max_indices = torch.argmax(x, dim=-1, keepdim=True)
    blank = torch.zeros_like(x)
    gathered = torch.gather(x, -1, max_indices)
    scattered = torch.scatter(blank, -1, max_indices, gathered)
    return x, max_indices, blank, gathered, scattered

def create_maxi(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        dtype: torch.dtype = torch.float32,
        dir: str = "data",
        name: str = "maxi",
):
    x, max_indices, blank, gathered, scattered = create_tensors(shape, dtype)
    save_reference(x, dir, f"{name}_x")
    save_reference(max_indices, dir, f"{name}_max_indices")
    save_reference(blank, dir, f"{name}_blank")
    save_reference(gathered, dir, f"{name}_gathered")
    save_reference(scattered, dir, f"{name}_scattered")

if __name__ == "__main__":
    # batch size, nodes, mems
    create_maxi((6, 7, 8), dtype=torch.float32, dir="data", name="maxi")